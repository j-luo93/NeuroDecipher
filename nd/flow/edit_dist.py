import logging

import numpy as np
import torch

import editdistance
from dev_misc import get_tensor


def compute_expected_edits(known_charset, log_probs, wordlist, valid_log_probs, num_samples=10, alpha=1e1, edit=False):
    logging.debug('Computing expected edits')
    log_probs = log_probs.transpose(0, 2).transpose(1, 2)  # size: bs x tl x C
    log_probs = torch.log_softmax(log_probs * alpha, dim=-1)
    probs = log_probs.exp()
    bs, tl, nc = probs.shape
    # get samples
    if num_samples > 0:
        samples = torch.multinomial(probs.reshape(bs * tl, nc), num_samples, replacement=True)
        samples = samples.view(bs, tl, num_samples)
        # get tokens
        tokens = known_charset.get_tokens(samples.transpose(1, 2))  # size: bs x num_samples
        # get probs
        sample_log_probs = log_probs[torch.arange(bs).long().view(-1, 1, 1),
                                     torch.arange(tl).long().view(1, -1, 1), samples]  # bs x tl x ns
        lengths = get_tensor(np.vectorize(len)(tokens) + 1, dtype='f')  # bs x num_samples
        mask = get_tensor(torch.arange(tl)).float().view(
            1, -1, 1).expand(bs, tl, num_samples) < lengths.unsqueeze(dim=1)
        sample_log_probs = (mask.float() * sample_log_probs).sum(dim=1)  # bs x num_samples
    else:  # This means we are taking the argmax according to token-level probs, not character-level probs.
        # Take argmax
        _, idx = valid_log_probs.max(dim=-1)
        tokens = wordlist[idx.cpu().numpy()].reshape(bs, 1)
        num_samples = 1
        sample_log_probs = get_tensor(np.ones([bs, 1]))
    # use chunks to get all edits
    chunk_size = 1000
    num_chunks = len(wordlist) // chunk_size + (len(wordlist) % chunk_size > 0)
    expected_edits = list()
    for i in range(num_chunks):
        logging.debug('Computing chunk %d/%d' % (i + 1, num_chunks))
        start = i * chunk_size
        end = min(start + chunk_size, len(wordlist))

        valid_log_prob_chunk = valid_log_probs[:, start: end]
        if edit:
            # get dists
            dists = compute_dists(tokens, wordlist[start: end])  # bs x c_s x (1 + ns)
            dists = get_tensor(dists, 'f')
            # remove accidental hits
            duplicates = compute_duplicates(tokens, wordlist[start: end])  # bs x c_s x (1 + ns)
            duplicates = get_tensor(duplicates, 'f')
            edit_chunk = dists * duplicates
            # compute expected edits
            ex_sample_log_probs = sample_log_probs.view(
                bs, 1, num_samples).expand(-1, valid_log_prob_chunk.shape[-1], -1)
            all_sample_log_probs = torch.cat([valid_log_prob_chunk.unsqueeze(dim=-1), ex_sample_log_probs], dim=-1)
            # make it less sharp
            all_sample_log_probs = all_sample_log_probs + (1.0 - duplicates) * (-999.)
            logits = all_sample_log_probs  # * alpha
            sm_log_probs = torch.log_softmax(logits, dim=-1)  # NOTE sm stands for softmax
            sm_probs = sm_log_probs.exp()
            expected_edits.append((edit_chunk * sm_probs).sum(dim=-1))
            # expected_edits.append(dists[..., 1:].sum(dim=-1))
        else:
            expected_edits.append(-valid_log_prob_chunk.tensor)
    return torch.cat(expected_edits, dim=1)


def compute_dists(sample_forms, wordlist):
    # global _DISTS_CACHE
    bs, ns = sample_forms.shape
    sample_forms = sample_forms.flatten()
    # dists = np.zeros([bs, len(wordlist), 1 + ns]) # NOTE always include the true tokens
    edits = editdistance.eval_all(wordlist, sample_forms)  # len(wl) x (bs x ns)
    edits = edits.reshape(len(wordlist), bs, ns)
    dists = np.transpose(edits, [1, 0, 2])
    dists = np.concatenate([np.zeros([bs, len(wordlist), 1], dtype='int64'), dists], axis=-1)
    dists = dists.astype('float32')
    # for i, b_samples in enumerate(sample_forms):
    #    for j, orig in enumerate(wordlist):
    #        for k, b_sample in enumerate(b_samples, 1): # NOTE starting with 1 -- 0 is reserved for the true tokens
    #            if (b_sample, orig) not in _DISTS_CACHE:
    #                _DISTS_CACHE[(b_sample, orig)] = editdistance.eval(b_sample, orig)
    #            d = _DISTS_CACHE[(b_sample, orig)]
    #            dists[i, j, k] = d
    # #dists += 1 # NOTE make sure they're all nonnegative, and even the ground truth has one unit of cost
    lengths = np.asarray(list(map(len, wordlist)))
    sample_lengths = np.asarray(list(map(len, sample_forms))).reshape(bs, ns)
    min_lengths = np.minimum(lengths.reshape(1, -1, 1), sample_lengths.reshape(bs, 1, ns))
    min_lengths = np.concatenate([np.repeat(lengths.reshape(1, -1), bs, axis=0).reshape(bs, -1, 1),
                                  min_lengths], axis=-1) + 1  # NOTE add one to avoid divide-by-zero error
    dists = dists / min_lengths
    # dists = dists ** 2
    return dists


def compute_duplicates(sample_forms, wordlist):
    bs, ns = sample_forms.shape
    dups = np.ones([bs, len(wordlist), 1 + ns])
    for i, b_samples in enumerate(sample_forms):
        sampled = {}
        # remove duplicated within the samples
        for k, b_sample in enumerate(b_samples, 1):
            if b_sample in sampled:
                dups[i, :, k] = 0.0
                continue
            sampled[b_sample] = k
        # remove samples that are identical to wordlist
        for j, orig in enumerate(wordlist):
            if orig in sampled:
                k = sampled[orig]
                dups[i, j, k] = 0.0
    return dups
