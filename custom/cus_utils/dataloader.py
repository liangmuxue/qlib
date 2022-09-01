class data():
    def __init__(self):
        self.idx = None # input index
        self.x0 = [[]] # string input, raw sentence
        self.x1 = [[]] # string input, tokenized
        self.xc = [[]] # indexed input, character-level
        self.xw = [[]] # indexed input, word-level
        self.y0 = [[]] # actual output
        self.y1 = None # predicted output
        self.prob = None # probability
        self.attn = None # attention heatmap

    def sort(self):
        self.idx = list(range(len(self.xw)))
        self.idx.sort(key = lambda x: -len(self.xw[x]))
        xc = [self.xc[i] for i in self.idx]
        xw = [self.xw[i] for i in self.idx]
        y0 = [self.y0[i] for i in self.idx]
        lens = list(map(len, xw))
        return xc, xw, y0, lens

    def unsort(self):
        self.idx = sorted(range(len(self.x0)), key = lambda x: self.idx[x])
        self.y1 = [self.y1[i] for i in self.idx]
        if self.prob: self.prob = [self.prob[i] for i in self.idx]
        if self.attn: self.attn = [self.attn[i] for i in self.idx]


