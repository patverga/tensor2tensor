from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from shutil import copyfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID





@registry.register_problem
class TranslateUschemaFb15k(translate.TranslateProblem):

  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateUschemaFb15k, self).__init__(was_reversed, was_copy)
    self.root_dir = '/iesl/data/clueweb_2016_full/fb15k_subset/kb/tmp'


  @property
  def targeted_vocab_size(self):
    return 90389
    # return 2**15  # 8192

  @property
  def vocab_name(self):
    return 'vocab.uschema_fb15k_all'

  def generator(self, data_dir, tmp_dir, train):
    tag = "train" if train else "dev"
    source_path = '%s/%s' % (self.root_dir, tag)

    vocab_file = '%s/vocab.uschema_fb15k_all' % self.root_dir
    symbolizer_vocab = text_encoder.TokenTextEncoder(vocab_file, replace_oov='<UNK>')

    return translate.token_generator(source_path + ".lang1", source_path + ".lang2",
                                     symbolizer_vocab, EOS)

  def feature_encoders(self, data_dir):

    vocab_filename = os.path.join(self.root_dir, self.vocab_name)
    token = text_encoder.TokenTextEncoder(vocab_filename, replace_oov='<UNK>')
    return {
      "inputs": token,
      "targets": token,
    }

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK

  @property
  def use_subword_tokenizer(self):
    return False

@registry.register_problem
class TranslateUschemaFb15kBoth(TranslateUschemaFb15k):


  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateUschemaFb15k, self).__init__(was_reversed, was_copy)
    self.root_dir = '/iesl/data/clueweb_2016_full/fb15k_subset/kb/tmp2'


  @property
  def targeted_vocab_size(self):
    return 90389
    # return 2**15  # 8192

  @property
  def vocab_name(self):
    return 'vocab.uschema_fb15k_all'

  def generator(self, data_dir, tmp_dir, train):
    root_dir = '/iesl/data/clueweb_2016_full/fb15k_subset/kb/tmp'
    tag = "train" if train else "dev"
    source_path = '%s/%s' % (root_dir, tag)

    vocab_file = '%s/vocab.uschema_fb15k_all.%d' % (root_dir, self.targeted_vocab_size)
    symbolizer_vocab = text_encoder.TokenTextEncoder(vocab_file, replace_oov='<UNK>')

    return translate.token_generator(source_path + ".lang1", source_path + ".lang2",
                                     symbolizer_vocab, EOS)



@registry.register_problem
class TranslateUschemaFreebase(TranslateUschemaFb15k):
  """Problem spec for WMT En-De translation."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateUschemaFreebase, self).__init__(was_reversed, was_copy)
    self.root_dir = '/iesl/data/clueweb_2016_full/freebase_ep_subset/2500k_ep_subset'

  @property
  def targeted_vocab_size(self):
    return 513227
    # return 2**15  # 8192

  @property
  def vocab_name(self):
    return 'vocab.uschema_fb'

  @property
  def source_vocab_name(self):
    return "%s_source" % self.vocab_name

  @property
  def target_vocab_name(self):
    return "%s_target" % self.vocab_name

  def generator(self, data_dir, tmp_dir, train):
    tag = "train" if train else "dev"
    source_path = '%s/%s' % (self.root_dir, tag)

    source_vocab_file = '%s/%s_source' % (self.vocab_name, self.root_dir)
    target_vocab_file = '%s/%s_kb_target' % (self.vocab_name, self.root_dir)
    source_vocab = text_encoder.TokenTextEncoder(source_vocab_file)
    target_vocab = text_encoder.TokenTextEncoder(target_vocab_file, replace_oov='<UNK>')

    return translate.bi_vocabs_token_generator(source_path + ".lang1", source_path + ".lang2",
                                               source_vocab, target_vocab, EOS)

  def feature_encoders(self, data_dir):

    source_vocab_filename = os.path.join(self.root_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(self.root_dir, self.target_vocab_name)
    source_token = text_encoder.TokenTextEncoder(source_vocab_filename)
    target_token = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov='<UNK>')
    return {
      "inputs": source_token,
      "targets": target_token,
    }


@registry.register_problem
class TranslateUschemaFreebaseKb(TranslateUschemaFreebase):
  """Problem spec for WMT En-De translation."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateUschemaFreebaseKb, self).__init__(was_reversed, was_copy)
    self.root_dir = '/iesl/data/clueweb_2016_full/freebase_ep_subset/2500k_ep_subset_with_kb'

  @property
  def vocab_name(self):
    return 'vocab.uschema_fb_kb'



@registry.register_problem
class TranslateUschemaFreebaseMedium(TranslateUschemaFreebase):
  """Problem spec for WMT En-De translation."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateUschemaFreebaseMedium, self).__init__(was_reversed, was_copy)
    self.root_dir = '/iesl/data/clueweb_2016_full/freebase_ep_subset/25k_entity_500k_ep'

  @property
  def vocab_name(self):
    return 'vocab.uschema_fb_medium'



