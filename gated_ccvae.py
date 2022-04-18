import os
import itertools
import sys
import numpy as np
import argparse
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from utils_data import CelebAReader, CELEBA_LABELS, CELEBA_EASY_LABELS
# from evaluation.eval import pred_binary_error
from keras.layers import Dense, Flatten, Add, Conv2D, Reshape, Conv2DTranspose
from utils import get_transn_loss, multi_sample_normal_np, kl_divergence_gaussian
# from utils.plot import plot_vae_loss
# from evaluation.eval import evaluate_model_discrete
from tensorflow_probability.python.distributions import Categorical, Normal

global file_txt_results_path
file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/temp_results.txt'


class Encoder(tf.keras.Model):
    def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
        self.z_dim = z_dim
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv4 = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv5 = Conv2D(filters=hidden_dim, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation='relu')
        self.flatten = Flatten()
        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, x):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv1(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv2(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv3(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.flatten(h)
        locs = self.locs_out(h)

        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        # Note if returning sampled z as well, decorate sample_fn with @tf.function so that it gets saved in the
        # computation graph and there is no need to implement it while testing or loading the model
        return locs, scale


class Decoder(tf.keras.Model):
    def __init__(self, hidden_dim=256, *args, **kwargs):
        super(Decoder, self).__init__()
        self.fc1 = Dense(units=hidden_dim, activation=tf.nn.relu)
        self.reshape = Reshape((1, 1, hidden_dim))
        self.conv1t = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation='relu')
        self.conv2t = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv3t = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv4t = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv5t = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid')

    def call(self, z):
        h = self.fc1(z)
        h = self.reshape(h)
        h = self.conv1t(h)
        h = self.conv2t(h)
        h = self.conv3t(h)
        h = self.conv4t(h)
        x = self.conv5t(h)
        return x


class Classifier(tf.keras.Model):
    def __init__(self, y_dim):
        super(Classifier, self).__init__()
        # Defining a dense layer will give many-to-one mapping from z to y. CCVAE used one-to-one mapping.
        self.out_prob_y = Dense(units=y_dim, activation=tf.nn.softmax)

    def call(self, encodes_z):
        prob_y = self.out_prob_y(encodes_z)
        return prob_y


class Conditional_Prior(tf.keras.Model):
    def __init__(self, z_dim):
        super(Conditional_Prior, self).__init__()
        # Defining a dense layer will give many-to-one mapping from y to z. CCVAE used one-to-one mapping.
        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, y, k=None):
        if not k:
            k = 1
        locs = self.locs_out(y)
        scale = self.std_out(y)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        prior_z_y = Normal(loc=locs, scale=scale)
        return locs, scale, prior_z_y.sample(sample_shape=k)


class CCVAE(tf.keras.Model):
    def __init__(self, z_dim, z_classify, y_dim):
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

        # Gating Parameters - Declare a tensor of variable parameters
        self.mu = tf.Variable(initial_value=tf.constant(0.5, shape=(y_dim, z_classify)), trainable=True)

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

    def sample_gating_parameter(self, mu, temperature, EPSILON=1e-30):
        mu = tf.clip_by_value(mu, clip_value_min=0.0, clip_value_max=1.0)
        eps1 = self.sample_gumbel_tf(tf.shape(mu))
        eps2 = self.sample_gumbel_tf(tf.shape(mu))
        num = tf.exp((eps2 - eps1) / temperature)
        t1 = tf.pow(mu, 1. / temperature)
        t2 = tf.pow((1. - mu), 1. / temperature) * num
        c = t1 / (t1 + t2 + EPSILON)
        return c


class MyModel:
    def __init__(self, ip_shape, z_dim, z_classify, y_dim, num_samples, supervision, train_config):
        self.train_config = train_config
        self.ip_shape = ip_shape
        self.z_dim = z_dim
        self.z_classify = z_classify
        self.z_style = z_dim - z_classify
        self.y_dim = y_dim
        print("Input Shape:", ip_shape, " Latent Dim (z_classify = {}): ".format(z_classify), z_dim, " Label Dim:", y_dim,
              file=open(file_txt_results_path, 'a'))

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
        # their inital value will be used when tf.function traces the computation graph, in order to make sure that its
        # updated value is used either pass it as an arg to the function or declare it as a variable whose value can
        # then be changed outside using var.assign which will reflect automatically inside the computation graph
        self.p_Y = tf.Variable(1/y_dim, dtype=tf.float32)

        self.model = CCVAE(z_dim, z_classify, y_dim)
        self.optimiser = tf.keras.optimizers.Adam(self.lr)

    def load_model(self,  param_dir, model_id):
        model = CCVAE(self.z_dim, self.z_classify, self.y_dim)

        # TODO: Make necessary modifications so that tf saves the model and we can run it w/o declaring or building it
        # We have saved the weights of sub-classed models (not using Keras Sequential or Functional API),
        # therefore we need to pass input to build the variables first and then load the weights

        # BUILD First
        _ = model.encoder(np.ones([1, *self.ip_shape]), np.ones([1, self.z_dim]))
        _ = model.decoder(np.ones([1, self.z_dim]))
        _ = model.classifier(np.ones([1, self.z_classify]))
        _ = model.cond_prior(np.ones([1, self.y_dim]))

        model.encoder.load_weights(os.path.join(param_dir, "encoder_model_{}.h5".format(model_id)))
        model.decoder.load_weights(os.path.join(param_dir, "decoder_model_{}.h5".format(model_id)))
        model.classifier.load_weights(os.path.join(param_dir, "classifier_{}.h5".format(model_id)))
        model.cond_prior.load_weights(os.path.join(param_dir, "cond_prior_{}.h5".format(model_id)))

        return model

    def classifier_loss(self, x, y, k=100):
        [post_locs, post_scales] = self.model.encoder(x)
        post_locs = post_locs[:, self.z_style:]
        post_scales = post_scales[:, self.z_style:]
        # Draw k samples from q(z|x) and compute log(q(y_curr|z_k)) = log(q(y|z_k))*y_curr for each.
        qy_z_k = [(lambda _z:  tf.reduce_sum(tf.math.multiply(self.model.classifier(_z), y), axis=-1))(_z)
                  for _z in self.model.multi_sample_normal(post_locs, post_scales, self.z_classify, k)]
        qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
        lqy_x = tf.math.log(tf.reduce_sum(qy_z_k, axis=0) + self.eps) - tf.cast(tf.math.log(float(k)), dtype=tf.float32)
        return tf.reshape(lqy_x, shape=[tf.shape(x)[0], ])

    def unsup_loss(self, x):
        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(x)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # Split the z into z_style and z_classify
        z_style = z[:, :self.z_style]
        z_classify = z[:, self.z_style:]

        # INFERENCE: Compute the classification prob q(y|z)
        qy_z = self.model.classifier(z_classify)
        # Sample y
        sampled_curr_encode_y = self.model.gumbel_softmax_tf(qy_z, self.latent_sampler_temp, self.y_dim, is_prob=True)
        log_qy_z = tf.reduce_sum(tf.math.log(qy_z + self.eps) * sampled_curr_encode_y, axis=-1)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(tf.math.log(self.p_Y + self.eps) * sampled_curr_encode_y, axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y)
        [prior_locs, prior_scales, sampled_z_k] = self.model.cond_prior(sampled_curr_encode_y)

        # GENERATION: Compute the log-likelihood of actions with p(x|z), where z is sampled from q(z|x)
        p_actions = self.model.decoder(curr_state, z)

        # ELBO
        ll = tf.reduce_sum(next_action * tf.math.log(p_actions + self.eps), axis=1)
        kl = kl_divergence_gaussian(mu1=post_locs, log_sigma_sq1=tf.math.log(post_scales + self.eps),
                                    mu2=prior_locs, log_sigma_sq2=tf.math.log(prior_scales + self.eps),
                                    mean_batch=False)
        elbo = ll + log_py - kl - log_qy_z

        transn_loss = get_transn_loss(prev_encode, sampled_curr_encode_y)
        loss = tf.reduce_mean(-elbo)
        return loss + transn_loss

    def sup_loss(self, x, y):

        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(x)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # Split the z into z_style and z_classify
        z_style = z[:, :self.z_style]
        z_classify = z[:, self.z_style:]

        # INFERENCE: Compute the classification prob q(y|z)
        qy_z = self.model.classifier(z_classify)
        log_qy_z = tf.reduce_sum(tf.math.log(qy_z + self.eps) * y, axis=-1)

        # INFERENCE: Compute label classification q(y|x) <- Sum_z(q(y|z)*q(z|x)) ~ 1/k * Sum(q(y|z_k))
        log_qy_x = self.classifier_loss(x, y)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(tf.math.log(self.p_Y + self.eps) * curr_encode, axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y)
        [prior_locs, prior_scales, sampled_z_k] = self.model.cond_prior(y)

        # GENERATION: Compute the log-likelihood of actions i.e. p(x|z)
        p_actions = self.model.decoder(z)

        # We only want gradients wrt to params of qyz, so stop them propagating to qzx! Why? Ref. Appendix C.3.1
        # In short, to reduce the variance in the gradients of classifier param! To a certain extent these gradients can
        # be viewed as redundant, as there is already gradients to update the predictive distribution due to the
        # log q(y|x) term anyway

        # Note: PYTORCH Detach stops the tensor from being tracked in the subsequent operations involving the tensor:
        # The original implementation is detaching the tensor z
        # log_qy_z_ = tf.stop_gradient(tf.reduce_sum(tf.log(self.classifier([z]) + self.eps) * ln_curr_encode_y_ip,
        #                                            axis=-1))
        # Compute weighted ratio
        # w = tf.exp(log_qy_z_ - log_qy_x)
        w = tf.exp(log_qy_z - log_qy_x)

        # ELBO
        ll = tf.reduce_sum(next_action * tf.math.log(p_actions + self.eps), axis=1)
        kl = kl_divergence_gaussian(mu1=post_locs, log_sigma_sq1=tf.math.log(post_scales + self.eps),
                                    mu2=prior_locs, log_sigma_sq2=tf.math.log(prior_scales + self.eps),
                                    mean_batch=False)
        elbo_term1 = tf.math.multiply(w, ll - kl - log_qy_z)
        elbo = elbo_term1 + log_py + log_qy_x*self.alpha

        # Transition Loss does not make any sense since curr latent mode is already known
        # transn_loss = get_transn_loss(ln_prev_encode_y_ip, ln_curr_encode_y_ip)
        loss = tf.reduce_mean(-elbo)

        # For Debugging
        # self.obs_var1 = tf.reduce_mean(elbo_term1)
        # self.obs_var2 = tf.reduce_mean(log_qy_x*self.alpha)
        return loss

    # # # IMPORTANT
    # tf.function wraps the function for tf's graph computation [efficient, fast, portable].
    # It applies to a function and all other functions it calls.
    # No need for decorating fns that are called from inside train_step
    # # TIP
    # Include as much computation as possible under a tf.function to maximize the performance gain.
    # For example, decorate a whole training step or the entire training loop.
    @tf.function
    def train_step(self, x, y, supervised):
        with tf.GradientTape() as tape:
            if supervised:
                loss = self.sup_loss(x, y)
            else:
                loss = self.unsup_loss(x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, data_loaders, param_dir, fig_path, exp_num):

        # Train the Model
        for i in range(0, self.train_config['n_epochs']):

            # initialize variables to store loss values
            epoch_losses_sup = 0.0
            epoch_losses_unsup = 0.0

            # # # compute number of batches for an epoch
            if args.sup_frac == 1.0:  # fully supervised
                batches_per_epoch = np.ceil(data_loaders['sup'].n_s/self.train_config['batch_size'])
                period_sup_batches = 1
                sup_batches = batches_per_epoch
            elif args.sup_frac > 0.0:  # semi-supervised
                sup_batches = np.ceil(data_loaders['sup'].n_s/self.train_config['batch_size'])
                unsup_batches = np.ceil(data_loaders['unsup'].n_s/self.train_config['batch_size'])
                batches_per_epoch = sup_batches + unsup_batches
                period_sup_batches = int(batches_per_epoch / sup_batches)
            elif args.sup_frac == 0.0:  # unsupervised
                sup_batches = 0.0
                batches_per_epoch = np.ceil(data_loaders['unsup'].n_s/self.train_config['batch_size'])
                period_sup_batches = np.Inf
            else:
                assert False, "Data frac not correct"

            # setup the iterators for training data loaders
            if args.sup_frac != 0.0:
                sup_iter = iter(data_loaders["sup"].step())
            if args.sup_frac != 1.0:
                unsup_iter = iter(data_loaders["unsup"].step())

            # count the number of supervised batches seen in this epoch
            ctr_sup = 0

            for i in tqdm(range(batches_per_epoch)):
                # whether this batch is supervised or not
                is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches
                # extract the corresponding batch
                if is_supervised:
                    (xs, ys) = next(sup_iter)
                    ctr_sup += 1
                else:
                    (xs, ys) = next(unsup_iter)

                if is_supervised:
                    loss = self.train_step(xs, ys, supervised=True)
                    epoch_losses_sup += loss.numpy()
                else:
                    loss = self.train_step(xs, ys, supervised=False)
                    epoch_losses_unsup += loss.numpy()

            if self.train_config['perc_supervision']:
                validation_accuracy = cc_vae.accuracy(data_loaders['valid'])
            else:
                validation_accuracy = np.nan

        #     with torch.no_grad():
        #         # save some reconstructions
        #         img = CELEBACached.fixed_imgs
        #         if args.cuda:
        #             img = img.cuda()
        #         recon = cc_vae.reconstruct_img(img).view(-1, *im_shape)
        #         save_image(make_grid(recon, nrow=8), './data/output/recon.png')
        #         save_image(make_grid(img, nrow=8), './data/output/img.png')
        #
        #     print("[Epoch %03d] Sup Loss %.3f, Unsup Loss %.3f, Val Acc %.3f" %
        #           (epoch, epoch_losses_sup, epoch_losses_unsup, validation_accuracy))
        # cc_vae.save_models(args.data_dir)
        # test_acc = cc_vae.accuracy(data_loaders['test'])
        # print("Test acc %.3f" % test_acc)
        # cc_vae.latent_walk(img[5], './data/output')
        # return

            #     # Save the model
        #     avg_epoch_loss = np.average(np.array(epoch_loss))
        #     if avg_epoch_loss < max_loss:
        #         max_loss = avg_epoch_loss
        #         self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_best.h5"), overwrite=True)
        #         self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_best.h5"), overwrite=True)
        #         self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_best.h5"), overwrite=True)
        #         self.model.classifier.save_weights(os.path.join(param_dir, "classifier_best.h5"), overwrite=True)
        #
        #     if i == 0 or i == self.train_config['n_epochs']-1:
        #         self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_%d.h5" % i), overwrite=True)
        #         self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_%d.h5" % i), overwrite=True)
        #         self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_%d.h5" % i), overwrite=True)
        #         self.model.classifier.save_weights(os.path.join(param_dir, "classifier_%d.h5" % i), overwrite=True)
        #
        #     pbar.refresh()
        #     pbar.set_description("Epoch {}".format(i + 1))
        #     pbar.set_postfix(Loss=avg_epoch_loss)
        #     pbar.update(1)
        #
        # plot_vae_loss(sup_loss, fig_path + "_SupLoss", exp_num)
        # plot_vae_loss(unsup_loss, fig_path + "_UnsupLoss", exp_num)


def run(args, exp_num=0):

    print("\n\n---------------------------------- Supervision {} ----------------------------------".format(args.sup),
          file=open(file_txt_results_path, 'a'))

    train_config = {
        "n_epochs": args.n,
        "batch_size": args.bs,
        "num_iters": 10,
        "lr": args.lr,
        "init_temp": 0.1,
        "anneal_rate": 0.00003,
        'perc_supervision': args.sup,
        'z_dim': args.z_dim,
        'n_classes': len(CELEBA_EASY_LABELS),
    }

    im_shape = (64, 64, 3)

    # ################################################ Specify Paths ################################################ #
    # Root Directory
    root_dir = "./data"

    # Other Directory
    model_dir = os.path.join(root_dir, "models")

    # Model Directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('CCVAE'))

    # ################################################ Load Data ################################################ #
    print("Loading data", file=open(file_txt_results_path, 'a'))
    reader = CelebAReader(root_dir, train_config['perc_supervision'], train_config['batch_size'])
    loaders = reader.setup_data_loaders()

    # ################################################ Train Model ################################################ #
    if train_config['perc_supervision'] == 1.:
        num_samples = loaders['sup'].n_s
    elif train_config['perc_supervision'] == 0.:
        num_samples = loaders['unsup'].n_s
    else:
        num_samples = loaders['sup'].n_s + loaders['unsup'].n_s

    ssvae_learner = MyModel(ip_shape=im_shape, z_dim=train_config['z_dim'], z_classify=train_config['n_classes'],
                            y_dim=train_config['n_classes'], num_samples=num_samples,
                            supervision=train_config['perc_supervision'], train_config=train_config)
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
        ssvae_learner.train(demos_un, demos_ln, param_dir, fig_path, exp_num)
        print("Finish.", file=open(file_txt_results_path, 'a'))
    else:
        print("Model already exists! Skipping training", file=open(file_txt_results_path, 'a'))

    # ################################################ Test Model ################################################ #
    # model_id = 'best'  # best, 999
    # with open(test_data_path, 'rb') as f:
    #     test_traj_sac = pkl.load(f)
    #
    # print("\nTesting Data Results", file=open(file_txt_results_path, 'a'))
    # ssvae_learner = MyModel(state_dim, stack_state_dim, action_dim, z_dim, y_dim, num_samples, sup,
    #                         train_config)
    # ssvae_learner.load_model(param_dir, model_id)
    # evaluate_model_discrete(ssvae_learner.model, "vae", test_traj_sac, train_config, file_txt_results_path)


def parser_args(parser):
    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=200, type=int,
                        help="number of epochs to run")
    parser.add_argument('-sup', '--sup-frac', default=1.0,
                        type=float, help="supervised fractional amount of the data i.e. "
                                         "how many of the images have supervised labels."
                                         "Should be a multiple of train_size / batch_size")
    parser.add_argument('-zd', '--z_dim', default=45, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data path')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    run(args)
