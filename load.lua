--[[
Code for loading & analyzing saved model
--]]

dofile("word2vec.lua")
require 'cunn'

-- Default configuration
config = {}
config.corpus = "corpus.txt" -- input data
config.window = 5 -- (maximum) window size
config.dim = 100 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 10 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 3 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first

cmd = torch.CmdLine()
cmd:option("--snapshot", '', 'snapshot to load model from')
cmd:option("--model_info", '', 'model info such as vocab, word2index, etc')
params = cmd:parse(arg)

-- Load model
local model_info = torch.load(params.model_info)
local m = Word2Vec(config)
m:load(model_info, params.snapshot)

m:print_sim_words({"the","he","can"},5)
local vec = m:get_vector("quit")
print(vec)
