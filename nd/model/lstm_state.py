import torch

from dev_misc import get_zeros

'''
Just a wrapper of LSTM states: a list of (h, c) tuples
'''


class LSTMState(object):

    def __init__(self, states):
        self.states = states

    @classmethod
    def from_pytorch(cls, states):
        hs, cs = states
        _, bs, d = hs.shape
        hs = hs.view(-1, 2, bs, d)
        cs = cs.view(-1, 2, bs, d)
        nl = hs.shape[0]
        states = [(h.sum(dim=0), c.sum(dim=0)) for h, c in zip(hs.unbind(dim=0), cs.unbind(dim=0))]
        return LSTMState(states)

    @classmethod
    def stack(cls, iterator_states, dim):
        nl = len(iterator_states[0])
        hs = [list() for _ in range(nl)]
        cs = [list() for _ in range(nl)]
        for states in iterator_states:
            for i, state in enumerate(states.states):
                h, c = state
                hs[i].append(h)
                cs[i].append(c)
        states = list()
        for i in range(nl):
            h = torch.stack(hs[i], dim)
            c = torch.stack(cs[i], dim)
            states.append((h, c))
        return LSTMState(states)

    @classmethod
    def zero_state(cls, num_layers, shape):
        states = list()
        for _ in range(num_layers):
            h = get_zeros(*shape)
            c = get_zeros(*shape)
            states.append((h, c))
        return LSTMState(states)

    @property
    def shape(self):
        return self.states[0][0].shape

    def clone(self):
        new_states = [(s[0].clone(), s[1].clone()) for s in self.states]
        return LSTMState(new_states)

    def dim(self):
        return self.states[0][0].dim()

    def unsqueeze(self, dim):
        new_states = [(s[0].unsqueeze(dim), s[1].unsqueeze(dim)) for s in self.states]
        return LSTMState(new_states)

    def view(self, *sizes):
        new_states = [(s[0].view(*sizes), s[1].view(*sizes)) for s in self.states]
        return LSTMState(new_states)

    def unbind(self, dim):
        n = self.states[0][0].shape[dim]
        ret = [list() for _ in range(n)]
        for s in self.states:
            h, c = s
            hs = h.unbind(dim)
            cs = c.unbind(dim)
            for i, (h, c) in enumerate(zip(hs, cs)):
                ret[i].append((h, c))
        ret = tuple(LSTMState(s) for s in ret)
        return ret

    def expand(self, *sizes):
        new_states = [(s[0].expand(*sizes), s[1].expand(*sizes)) for s in self.states]
        return LSTMState(new_states)

    def contiguous(self):
        states = list()
        for s in self.states:
            h = s[0].contiguous()
            c = s[1].contiguous()
            states.append((h, c))
        return LSTMState(states)

    def __len__(self):
        return len(self.states)

    def detach_(self):
        for s in self.states:
            s[0].detach_()
            s[1].detach_()

    def size(self):
        return self.states[0][0].size()

    def cat(self, other, dim):
        states = list()
        for s1, s2 in zip(self.states, other.states):
            h = torch.cat([s1[0], s2[0]], dim)
            c = torch.cat([s1[1], s2[1]], dim)
            states.append((h, c))
        return LSTMState(states)

    def __mul__(self, other):
        states = list()
        for s in self.states:
            h = s[0] * other
            c = s[1] * other
            states.append((h, c))
        return LSTMState(states)

    def __add__(self, other):
        states = list()
        for s1, s2 in zip(self.states, other.states):
            h = s1[0] + s2[0]
            c = s1[1] + s2[1]
            states.append((h, c))
        return LSTMState(states)

    def get_output(self):
        return self.states[-1][0]

    def get(self, ind):
        return self.states[ind]

    def __getitem__(self, key):
        states = list()
        for s in self.states:
            h = s[0][key]
            c = s[1][key]
            states.append((h, c))
        return LSTMState(states)

    def __setitem__(self, key, item):
        for s1, s2 in zip(self.states, item.states):
            s1[0][key] = s2[0]
            s1[1][key] = s2[1]
