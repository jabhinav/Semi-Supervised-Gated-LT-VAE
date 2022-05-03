import json
import os
import sys
import logging
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
from utils_data import CelebAReader, CELEBA_LABELS, CELEBA_EASY_LABELS
from utils import get_gaussian_kl_div, img_log_likelihood, get_gates
from networks import *
from configs import get_config
import pandas as pd

from tensorflow_probability.python.distributions import Categorical, Normal, Bernoulli

logging.basicConfig(filename="./logs", filemode='w', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CCVAE(tf.keras.Model):
    def __init__(self, z_dim, z_classify, y_dim, train_config):
        super(CCVAE, self).__init__()
        # # ------------------------------------------ Model Declaration ------------------------------------------ # #
        self.z_dim = z_dim
        self.z_classify = z_classify
        self.z_style = z_dim - self.z_classify
        self.y_dim = y_dim
        # Encode input into latent space
        self.encoder = Encoder(z_dim)
        # Decode latent space into output
        self.decoder = Decoder(hidden_dim=z_dim)
        # Classifier to classify encoded input, z into one of the labels, y
        self.classifier = Classifier(y_dim)
        # Conditional Prior to obtain Prior probability of z given y (Originally it was N(0, I))
        self.cond_prior = Conditional_Prior(z_classify)

        self.initialise_mu(train_config)

    def initialise_mu(self, train_config):
        if train_config['gate_type'] == 'learnable':
            logging.info("Initialising mu with fixed value (learnable)")
            mu_init = train_config['mu_init']
            mu_init = tf.constant(mu_init, dtype=tf.float32)
            self.mu = tf.Variable(initial_value=mu_init, trainable=True)
        elif train_config['gate_type'] == 'fixed' and train_config['gate_subtype'] == 'inferred':
            logging.info("Initialising mu with fixed value")
            mu_init = train_config['mu_init']
            mu_init = tf.constant(mu_init, dtype=tf.float32)
            self.mu = tf.Variable(initial_value=mu_init, trainable=False)
        elif train_config['gate_type'] == 'fixed' and train_config['gate_subtype'] == 'one-one':
            # Create a diagonal matrix of size z_classify x y_dim
            mu_init = tf.eye(self.z_classify, self.y_dim)
            mu_init = tf.constant(mu_init)
            self.mu = tf.Variable(initial_value=mu_init, trainable=False)
        else:
            raise ValueError('Invalid gate type/subtype: {}/{}'.format(train_config['gate_type'],
                                                                       train_config['gate_subtype']))

    def sample_gumbel_tf(self, shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def sample_gumbel_softmax_tf(self, logits, temperature):
        y = logits + self.sample_gumbel_tf(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    def gumbel_softmax_tf(self, logits, temperature, latent_dim, is_prob=False):
        """
        Returns: Discrete Sampler which returns a prob. distribution over discrete states. The returned prob dist.
        starts to peak at 1 class with decrease in temp. Case Study: For uniformly distributed logits(& say if temp
        is 0.1) then all the classes are equally likely to be sampled with softmax distribution peaking at one of them.

        General Case: Returned Samples are more likely the ones that carry high prob in the input distribution when
        temp starts going down.
        """
        # If input is a prob distribution, convert into logits
        if is_prob:
            logits = tf.math.log(logits)
        # While testing, we get the exact one-hot vector else we go with softmaxed vector to allow diff. during training
        y = self.sample_gumbel_softmax_tf(logits, temperature)
        return tf.reshape(y, shape=[-1, latent_dim])

    def sample_unit_normal(self, shape):
        epsilon = tf.random.normal(shape, mean=0.0, stddev=1.0, )
        return epsilon

    def sample_normal(self, mu, std, latent_dim):
        epsilon = tf.random.normal(tf.shape(std), mean=0.0, stddev=1.0, )
        z = mu + tf.math.multiply(std, epsilon)
        return tf.reshape(z, shape=[-1, latent_dim])

    def multi_sample_normal(self, mu, std, latent_dim, k=100):
        samples = []
        for _ in tf.range(k):
            z = self.sample_normal(mu, std, latent_dim)
            samples.append(z)
        return samples

    def sample_gating_parameter(self, mu, temperature, EPSILON=1e-20):
        mu = tf.clip_by_value(mu, clip_value_min=0.0, clip_value_max=1.0)
        eps1 = self.sample_gumbel_tf(tf.shape(mu))
        eps2 = self.sample_gumbel_tf(tf.shape(mu))
        num = tf.exp((eps2 - eps1) / temperature)
        t1 = tf.pow(mu, 1. / temperature)
        t2 = tf.pow((1. - mu), 1. / temperature) * num
        c = t1 / (t1 + t2 + EPSILON)

        return c


class Learner:
    def __init__(self, ip_shape, z_dim, z_classify, y_dim, num_samples, supervision, train_config):
        self.train_config = train_config
        self.ip_shape = ip_shape
        self.z_dim = z_dim
        self.z_classify = z_classify
        self.z_style = z_dim - z_classify
        self.y_dim = y_dim
        logger.info("Input Shape: {}, Latent Dim (z_classify = {}): {}, Label Dim: {}".format(ip_shape, z_classify, z_dim, y_dim))

        # Parameters
        self.supervision = supervision
        self.eps = 1e-20
        self.lr = train_config['lr']

        # Classification q(y|x) weight
        self.alpha = 0.1*num_samples

        # Other Param
        # self.latent_sampler_temp = latent_sampler_temp = self.model_config.init_temp
        self.latent_sampler_temp = self.train_config['init_temp']
        self.gating_sampler_temp = self.train_config['gating_init_temp']

        # To avoid unintended behaviours when variables declared outside the scope of tf.function are modified because
        # their initial value will be used when tf.function traces the computation graph, in order to make sure that its
        # updated value is used either pass it as an arg to the function or declare it as a variable whose value can
        # then be changed outside using var.assign which will reflect automatically inside the computation graph
        self.p_Y = tf.Variable(np.ones([1, len(CELEBA_EASY_LABELS)]) / 2., dtype=tf.float32, trainable=False)

        self.model = CCVAE(z_dim, z_classify, y_dim, train_config)
        self.optimiser = tf.keras.optimizers.Adam(self.lr)

    def load_model(self,  param_dir, model_id):
        logger.info("Loading model from {} of model_id {}".format(param_dir, model_id))

        # BUILD First
        _ = self.model.encoder(np.ones([1, *self.ip_shape]))
        _ = self.model.decoder(np.ones([1, self.z_dim]))
        _ = self.model.classifier(np.ones([1, self.z_classify]), np.ones([self.z_classify, self.y_dim])/2.)
        _ = self.model.cond_prior(np.ones([1, self.y_dim]), np.ones([self.y_dim, self.z_classify], dtype=np.float32)/2.)

        self.model.encoder.load_weights(os.path.join(param_dir, "encoder_model_{}.h5".format(model_id)))
        self.model.decoder.load_weights(os.path.join(param_dir, "decoder_model_{}.h5".format(model_id)))
        self.model.classifier.load_weights(os.path.join(param_dir, "classifier_{}.h5".format(model_id)))
        self.model.cond_prior.load_weights(os.path.join(param_dir, "cond_prior_{}.h5".format(model_id)))

        # Load gating parameters. Hacky way to do it. Override the value of the variables (not trainable)
        if self.train_config['gate_type'] == 'learnable':
            mu_init = np.load(os.path.join(param_dir, "learned_gating_matrix_{}.npy".format(model_id)))
            # Load the mu_init into the model's variables
            self.model.mu = tf.Variable(initial_value=mu_init, trainable=True)
            logging.info("Loaded learned mu")

    def classifier_loss(self, x, y, c, k=100):
        [post_locs, post_scales] = self.model.encoder(x)
        # Draw k samples from q(z|x) and compute log(q(y_curr|z_k)) = log(q(y|z_k))*y_curr for each.
        log_qy_zc_k = []
        for _ in range(k):
            z = self.model.sample_normal(post_locs, post_scales, self.z_dim)
            z_classify = z[:, self.z_style:]
            z_classify_tiled = tf.tile(tf.expand_dims(z_classify, axis=-1), multiples=[1, 1, self.y_dim])
            logits_y_zc = self.model.classifier(z_classify_tiled, c)
            qy_zc = Bernoulli(logits=logits_y_zc)
            log_qy_zc = tf.reduce_sum(qy_zc.log_prob(y), axis=-1)
            log_qy_zc = tf.expand_dims(log_qy_zc, axis=0)
            log_qy_zc_k.append(log_qy_zc)
        log_qy_zc_k = tf.concat(log_qy_zc_k, axis=0)
        lqy_x = tf.reduce_logsumexp(log_qy_zc_k, axis=0) - tf.cast(tf.math.log(float(k)), dtype=tf.float32)  ## This can go to nan
        return tf.reshape(lqy_x, shape=[tf.shape(x)[0], ])

    def unsup_loss(self, x):

        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(x)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # Split the z into z_style and z_classify
        z_classify = z[:, self.z_style:]

        # Sample gating parameters c [Sample the latent graph topology]. It has shape [z_classify, y_dim]
        c = self.model.sample_gating_parameter(self.model.mu, temperature=self.gating_sampler_temp)
        # if tf.math.reduce_any(tf.math.is_nan(c)):
        #     raise ValueError("c is nan")

        # Tile z_classify to match the shape of c; Tiling does output_dim[i] = input_dim[i] * multiples[i].
        # z_classify_tiled has shape [batch, z_classify, y_dim]
        z_classify_tiled = tf.tile(tf.expand_dims(z_classify, axis=-1), multiples=[1, 1, self.y_dim])

        # INFERENCE: Compute the classification prob q(y|z, c)
        logits_y_zc = self.model.classifier(z_classify_tiled, c)
        qy_zc = Bernoulli(logits=logits_y_zc)
        # Sample y
        y = qy_zc.sample()
        log_qy_zc = tf.reduce_sum(qy_zc.log_prob(y), axis=-1)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(Bernoulli(probs=tf.tile(self.p_Y, [x.shape[0], 1])).log_prob(y), axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y)
        y_tiled = tf.tile(tf.expand_dims(y, axis=-1), multiples=[1, 1, self.z_classify])
        y_tiled = tf.cast(y_tiled, tf.float64)
        [prior_locs, prior_scales] = self.model.cond_prior(y_tiled, c)
        prior_locs = tf.concat([tf.zeros(shape=[tf.shape(x)[0], self.z_style]), prior_locs], axis=-1)
        prior_scales = tf.concat([tf.ones(shape=[tf.shape(x)[0], self.z_style]), prior_scales], axis=-1)
        kl = get_gaussian_kl_div(post_locs, post_scales, prior_locs, prior_scales)

        # GENERATION: Compute the log-likelihood of reconstruction i.e. p(x|z)
        recon_x = self.model.decoder(z)
        log_pxz = img_log_likelihood(recon_x, x)

        # ELBO
        elbo = log_pxz + log_py - kl - log_qy_zc
        loss = tf.reduce_mean(-elbo)

        # Add L1 regularization to the parameters of gating variables
        if self.train_config['gate_type'] == 'learnable':
            loss += self.train_config['gating_reg'] * tf.reduce_mean(tf.abs(self.model.mu))

        return loss, c

    def sup_loss(self, x, y):

        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(x)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # Split the z into z_style and z_classify
        z_classify = z[:, self.z_style:]

        # Sample gating parameters c [Sample the latent graph topology]. It has shape [z_classify, y_dim]
        c = self.model.sample_gating_parameter(self.model.mu, temperature=self.gating_sampler_temp)

        # Tile z_classify to match the shape of c; Tiling does output_dim[i] = input_dim[i] * multiples[i].
        # z_classify_tiled has shape [batch, z_classify, y_dim]
        z_classify_tiled = tf.tile(tf.expand_dims(z_classify, axis=-1), multiples=[1, 1, self.y_dim])

        # INFERENCE: Compute the classification prob q(y|z,c)
        logits_y_zc = self.model.classifier(z_classify_tiled, c)
        qy_zc = Bernoulli(logits=logits_y_zc)
        log_qy_zc = tf.reduce_sum(qy_zc.log_prob(y), axis=-1)

        # INFERENCE: Compute label classification q(y|x) <- Sum_z(q(y|z)*q(z|x)) ~ 1/k * Sum(q(y|z_k))
        log_qy_x = self.classifier_loss(x, y, c)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(Bernoulli(probs=tf.tile(self.p_Y, [x.shape[0], 1])).log_prob(y), axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y) and then the kl divergence
        y_tiled = tf.tile(tf.expand_dims(y, axis=-1), multiples=[1, 1, self.z_classify])
        # Change the type of y_tiled to float32
        y_tiled = tf.cast(y_tiled, tf.float64)
        [prior_locs, prior_scales] = self.model.cond_prior(y_tiled, c)
        prior_locs = tf.concat([tf.zeros(shape=[tf.shape(x)[0], self.z_style]), prior_locs], axis=-1)
        prior_scales = tf.concat([tf.ones(shape=[tf.shape(x)[0], self.z_style]), prior_scales], axis=-1)
        kl = get_gaussian_kl_div(post_locs, post_scales, prior_locs, prior_scales)

        # GENERATION: Compute the log-likelihood of reconstruction i.e. p(x|z)
        recon_x = self.model.decoder(z)
        log_pxz = img_log_likelihood(recon_x, x)

        # We only want gradients wrt to params of qyz, so stop them propagating to qzx! Why? Ref. Appendix C.3.1
        # In short, to reduce the variance in the gradients of classifier param! To a certain extent these gradients can
        # be viewed as redundant, as there is already gradients to update the predictive distribution due to the
        # log q(y|x) term anyway

        # The original implementation is detaching the tensor z_classify
        z_classify_detached = tf.stop_gradient(z_classify)
        z_classify_tiled_ = tf.tile(tf.expand_dims(z_classify_detached, axis=-1), multiples=[1, 1, self.y_dim])
        logits_y_zc_ = self.model.classifier(z_classify_tiled_, c)
        qy_zc_ = Bernoulli(logits=logits_y_zc_)
        log_qy_zc_ = tf.reduce_sum(qy_zc_.log_prob(y), axis=-1)

        # Compute weighted ratio
        w = tf.exp(log_qy_zc_ - log_qy_x)
        # w = tf.exp(log_qy_zc - log_qy_x)

        # ELBO
        elbo_term1 = tf.math.multiply(w, log_pxz - kl - log_qy_zc)
        # elbo = elbo_term1 + log_py + log_qy_x*self.alpha
        elbo = elbo_term1 + log_py + log_qy_x
        loss = tf.reduce_mean(-elbo)

        # Add L1 regularization to the parameters of gating variables
        if self.train_config['gate_type'] == 'learnable':
            loss += self.train_config['gating_reg'] * tf.reduce_mean(tf.abs(self.model.mu))

        return loss, c

    @tf.function
    def train_step(self, x, y, supervised):
        with tf.GradientTape() as tape:
            if supervised:
                loss, c = self.sup_loss(x, y)
            else:
                loss, c = self.unsup_loss(x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, c

    def train(self, data_loaders, param_dir, fig_path):

        best_val_acc = -np.inf
        # Train the Model
        for epoch in range(0, self.train_config['n_epochs']):

            # compute number of batches for an epoch
            if self.train_config['perc_supervision'] == 1.0:  # fully supervised
                batches_per_epoch = np.ceil(data_loaders['sup'].n_s/self.train_config['batch_size'])
                period_sup_batches = 1
                sup_batches = batches_per_epoch
            elif self.train_config['perc_supervision'] > 0.0:  # semi-supervised
                sup_batches = np.ceil(data_loaders['sup'].n_s/self.train_config['batch_size'])
                unsup_batches = np.ceil(data_loaders['unsup'].n_s/self.train_config['batch_size'])
                batches_per_epoch = sup_batches + unsup_batches
                period_sup_batches = int(batches_per_epoch / sup_batches)
            elif self.train_config['perc_supervision'] == 0.0:  # unsupervised
                sup_batches = 0.0
                batches_per_epoch = np.ceil(data_loaders['unsup'].n_s/self.train_config['batch_size'])
                period_sup_batches = np.Inf
            else:
                assert False, "Data frac not correct"

            # setup the iterators for training data loaders
            if self.train_config['perc_supervision'] != 0.0:
                sup_iter = iter(data_loaders["sup"].step())
            if self.train_config['perc_supervision'] != 1.0:
                unsup_iter = iter(data_loaders["unsup"].step())

            # count the number of supervised batches seen in this epoch
            ctr_sup = 0

            # initialize variables to store loss values
            epoch_losses_sup = 0.0
            epoch_losses_unsup = 0.0
            sup_loss, unsup_loss = 0.0, 0.0

            # # # TRAINING LOOP
            with tqdm(total=int(batches_per_epoch)) as pbar:
                for i in range(int(batches_per_epoch)):
                    # whether this batch is supervised or not
                    is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches
                    # extract the corresponding batch
                    if is_supervised:
                        (xs, ys) = next(sup_iter)
                        ctr_sup += 1
                    else:
                        (xs, ys) = next(unsup_iter)

                    if is_supervised:
                        sup_loss, c = self.train_step(xs, ys, supervised=True)
                        sup_loss = sup_loss.numpy()
                        # epoch_losses_sup += sup_loss.numpy()
                    else:
                        unsup_loss, c = self.train_step(xs, ys, supervised=False)
                        unsup_loss = unsup_loss.numpy()
                        # epoch_losses_unsup += unsup_loss.numpy()

                    c_total = np.sum(c.numpy())
                    # Check if c is nan
                    if np.isnan(c.numpy()).any():
                        print(c.numpy())
                        sys.exit(-1)
                    pbar.refresh()
                    pbar.set_description("Iteration: {}, Epoch: {}".format(i + 1, epoch + 1))
                    pbar.set_postfix(SupCtr=ctr_sup, SupLoss=sup_loss, UnsupLoss=unsup_loss, c_avg=c_total)
                    pbar.update(1)

            if self.train_config['perc_supervision']:
                validation_accuracy = self.accuracy(data_loaders['valid'])
            else:
                validation_accuracy = -np.inf

            logger.info("[Epoch %03d] Val Acc %.3f" % (epoch, validation_accuracy))

            if validation_accuracy > best_val_acc:
                logger.info("Saving best model...")
                best_val_acc = validation_accuracy
                self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_best.h5"), overwrite=True)
                self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_best.h5"), overwrite=True)
                self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_best.h5"), overwrite=True)
                self.model.classifier.save_weights(os.path.join(param_dir, "classifier_best.h5"), overwrite=True)
                if self.train_config['gate_type'] == 'learnable':
                    np.save(os.path.join(param_dir, "learned_gating_matrix_best.npy"), self.model.mu.numpy())

                    # Save in Pandas format
                    indexes = ["z{}".format(i + 1) for i in range(len(CELEBA_EASY_LABELS))]
                    gating_df = pd.DataFrame(self.model.mu.numpy(), index=indexes, columns=CELEBA_EASY_LABELS)
                    gating_df.to_csv(os.path.join(param_dir, "learned_gating_matrix_best.csv"))

            # Update the temperature of the gating network
            if self.train_config['gate_type'] == 'learnable':
                self.gating_sampler_temp *= 0.99
                logger.info('gating_sampler_temp decayed to: %.4f' % self.gating_sampler_temp)

        # Save the model
        logger.info("Saving last model...")
        self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_last.h5"), overwrite=True)
        self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_last.h5"), overwrite=True)
        self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_last.h5"), overwrite=True)
        self.model.classifier.save_weights(os.path.join(param_dir, "classifier_last.h5"), overwrite=True)
        if self.train_config['gate_type'] == 'learnable':
            np.save(os.path.join(param_dir, "learned_gating_matrix_last.npy"), self.model.mu.numpy())
            # Save in Pandas format
            indexes = ["z{}".format(i + 1) for i in range(len(CELEBA_EASY_LABELS))]
            gating_df = pd.DataFrame(self.model.mu.numpy(), index=indexes, columns=CELEBA_EASY_LABELS)
            gating_df.to_csv(os.path.join(param_dir, "learned_gating_matrix_last.csv"))

    def classifier_accuracy(self, x, y):
        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(x)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # Split the z into z_style and z_classify
        z_classify = z[:, self.z_style:]

        # Sample gating parameters c [Sample the latent graph topology]. It has shape [z_classify, y_dim]
        c = self.model.sample_gating_parameter(self.model.mu, temperature=self.gating_sampler_temp)

        # Tile z_classify to match the shape of c; Tiling does output_dim[i] = input_dim[i] * multiples[i].
        # z_classify_tiled has shape [batch, z_classify, y_dim]
        z_classify_tiled = tf.tile(tf.expand_dims(z_classify, axis=-1), multiples=[1, 1, self.y_dim])

        # INFERENCE: Compute the classification prob q(y|z, c)
        logits_y_zc = self.model.classifier(z_classify_tiled, c)
        # Convert logits to probabilities using sigmoid
        probs_y_zc = tf.sigmoid(logits_y_zc)
        # Round off the probabilities to get the class labels
        y_hat = tf.round(probs_y_zc)
        # Compare the preidicted labels with the true labels
        correct_prediction = tf.equal(y_hat, y)
        # Compute the accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def accuracy(self, data_loader):
        acc = 0.0
        iterator = iter(data_loader.step())
        num_batches = np.ceil(data_loader.n_s/self.train_config['batch_size'])
        for i in range(int(num_batches)):
            (xs, ys) = next(iterator)
            acc += self.classifier_accuracy(xs, ys)
        return acc / num_batches


def run(args, sup=0.0):

    logger.info("\n\n---------------------------------- Supervision {} ----------------------------------".format(sup))

    train_config = {
        "n_epochs": args.n,
        "batch_size": args.bs,
        "num_iters": 10,
        "lr": args.lr,
        "init_temp": 0.1,
        "anneal_rate": args.anneal_rate,
        'perc_supervision': sup,
        'z_dim': args.z_dim,
        'n_classes': len(CELEBA_EASY_LABELS),
        'gate_type': args.gate_type,
        'gate_subtype': args.gate_subtype,
        'gating_init_temp': 1.0 if args.gate_type == 'learnable' else 0.3,
        'gating_reg': args.l1_reg
    }

    # Dump the config into log file
    logger.info(json.dumps(train_config, indent=4))

    im_shape = (64, 64, 3)

    # ################################################ Specify Paths ################################################ #
    # Root Directory
    root_dir = args.data_dir

    # Data Directory
    data_dir = os.path.join(root_dir, "data")

    # Model Directory
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.gate_type == 'learnable':
        param_dir = os.path.join(model_dir, "params_{}_{}".format(sup, args.gate_type))
    else:
        param_dir = os.path.join(model_dir, "params_{}_{}_{}".format(sup, args.gate_type, args.gate_subtype))

    fig_path = os.path.join(model_dir, '{}_loss'.format('CCVAE'))

    # ################################################ Load Data ################################################ #
    logger.info("Loading data...")
    reader = CelebAReader(data_dir, train_config['perc_supervision'], train_config['batch_size'])
    loaders = reader.setup_data_loaders()

    mu = reader.init_gating_prob
    train_config['mu_init'] = mu

    # ################################################ Train Model ################################################ #
    if train_config['perc_supervision'] == 1.:
        num_samples = loaders['sup'].n_s
    elif train_config['perc_supervision'] == 0.:
        num_samples = loaders['unsup'].n_s
    else:
        num_samples = loaders['sup'].n_s + loaders['unsup'].n_s

    ssvae_learner = Learner(ip_shape=im_shape, z_dim=train_config['z_dim'], z_classify=train_config['n_classes'],
                            y_dim=train_config['n_classes'], num_samples=num_samples,
                            supervision=train_config['perc_supervision'], train_config=train_config)
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    if args.do_train:
        logger.info("Training model..")
        ssvae_learner.train(loaders, param_dir, fig_path)
        logger.info("Finish Training")
    else:
        logger.info("Skipping training")

    # ################################################ Test Model ################################################ #
    if args.do_test:
        logger.info("Testing best model..")
        ssvae_learner.load_model(param_dir, 'best')
        # Since we are testing, set the gating sampler temperature to 0.3
        ssvae_learner.gating_sampler_temp = 0.3  # This stmt. will only affect the gating scheme = 'learnable'
        test_accuracy = ssvae_learner.accuracy(loaders['test'])
        logger.info("Test Accuracy (best model): %.3f" % test_accuracy)


if __name__ == "__main__":

    args = get_config()

    sup = [1.0, 0.5, 0.2]
    for s in sup:
        run(args, sup=s)
