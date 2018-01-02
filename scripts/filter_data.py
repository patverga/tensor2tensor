import hashlib
import gzip
import glob
import sys
import os

data_root = '/iesl/data/clueweb_2016_full/freebase_ep_subset'
ep_f = '%s/stats/top_2500k_eps.counts' % data_root
out_dir = '%s/2500k_ep_subset' % data_root
out_f = '%s/2500k_subset' % out_dir
# take this many examples per ep
example_k = 10

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print ('Reading in ep subset from %s' % ep_f)
# read in the set of eps we want to keep and initialize their counts to 0
with open(ep_f, 'r') as f:
    ep_set = {l.strip().split('\t')[0].replace('::', ' '): 0 for l in f}

used_sentences = set()
in_f_pattern = '%s/*gz' % data_root
in_files = glob.glob(in_f_pattern)
exported_n = 0
print('Subsetting data from %s and writing to %s' % (in_f_pattern, out_f))
with open('%s.lang1' % out_f, 'w') as source_out_f:
    with open('%s.lang2' % out_f, 'w') as target_out_f:
        for file_num, in_f in enumerate(in_files):
            with gzip.open(in_f, 'r') as f:
                print('\n%s\n' % in_f)
                for line_num, line in enumerate(f):
                    if line_num % 100000 == 0:
                        sys.stdout.write('\r file: %d line: %2.1fM wrote: %2.1fK'
                                         % (file_num, (line_num/1000000.), exported_n/1000.))
                        sys.stdout.flush()
                    # parse next line
                    try:
                        e1, _, mention_1, start_1, end_1, e2, _, mention_2, start_2, end_2, doc_id, _, sentence \
                            = line.strip().split('\t')
                        ep = '%s %s' % (e1, e2)
                        # only keep eps in subset and sample the first k unique sentences from that ep
                        if ep in ep_set and ep_set[ep] < example_k:
                            sentence = sentence.split()
                            # put arg tokens into sentences
                            s1 = int(start_1)
                            s2 = int(start_2)
                            e1 = int(end_1)
                            e2 = int(end_2)
                            if s1 < s2:
                                first_arg = ['$ARG1']
                                second_arg = ['$ARG2']
                                left = sentence[:s1]
                                center = sentence[e1:s2]
                                right = sentence[e2:]
                            else:
                                first_arg = ['$ARG2']
                                second_arg = ['$ARG1']
                                left = sentence[:s2]
                                center = sentence[e2:s1]
                                right = sentence[e1:]
                            arg_sentence = ' '.join(left + first_arg + center + second_arg + right)

                            # check that this ep/sentence has not been sampled yet
                            example_hash = hashlib.sha1('%s_%s_%s' % (e1, e2, arg_sentence)).hexdigest()
                            if example_hash not in used_sentences:
                                ep_set[ep] += 1
                                exported_n += 1
                                used_sentences.add(example_hash)
                                source_out_f.write('%s\n' % ep)
                                target_out_f.write('%s\n' % arg_sentence)
                    except Exception as e:
                        print(e)
print('\n Done')
