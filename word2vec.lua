--[[
Class for word2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")
require 'cunn'
require 'util'

local tds = require 'tds'
local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
  self.tensortype = torch.getdefaulttensortype()
  self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
  self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
  self.neg_samples = config.neg_samples
  self.minfreq = config.minfreq
  self.dim = config.dim
  self.criterion = nn.BCECriterion() -- logistic loss
  self.word = torch.IntTensor(1) 
  self.contexts = torch.IntTensor(1+self.neg_samples) 
  self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
  self.window = config.window 
  self.lr = config.lr 
  self.min_lr = config.min_lr
  self.alpha = config.alpha
  self.table_size = config.table_size 
  self.vocab = tds.hash()
  self.index2word = tds.hash()
  self.word2index = tds.hash()
  self.total_count = 0
end

function Word2Vec:load_model_info(model_info)
  model_info = torch.load(model_info)
  self.vocab = model_info.vocab
  self.index2word = model_info.index2word
  self.vocab_size = #self.index2word
  self.word2index = model_info.word2index
  self.table = model_info.table
  self.total_count = model_info.total_count
end

function Word2Vec:load(model_info, snapshot, gpu_no)
  self:load_model_info(model_info)
  
  cutorch.setDevice(gpu_no + 1)

  self.w2v = torch.load(snapshot) 
  self.context_vecs = self.w2v.modules[1].modules[1]
  self.word_vecs = self.w2v.modules[1].modules[2]
end

-- move to cuda
function Word2Vec:cuda()
  self.word = self.word:cuda()
  self.contexts = self.contexts:cuda()
  self.labels = self.labels:cuda()
  self.criterion:cuda()
  self.w2v:cuda()
end

-- Build vocab frequency, word2index, and index2word from input file
function Word2Vec:build_vocab(corpus)
  print_f("Building vocabulary...")
  local start = sys.clock()
  local f = io.open(corpus, "r")
  local n = 1
  for line in f:lines() do
    for _, word in ipairs(self:split(line)) do
      self.total_count = self.total_count + 1
      if self.vocab[word] == nil then
        self.vocab[word] = 1	 
      else
        self.vocab[word] = self.vocab[word] + 1
      end
    end
    n = n + 1
  end
  f:close()
  -- Delete words that do not meet the minfreq threshold and create word indices
  for word, count in pairs(self.vocab) do
    if count >= self.minfreq then
      self.index2word[#self.index2word+1] = word
      self.word2index[word] = #self.index2word	    
    else
      self.vocab[word] = nil
    end
  end
  self.vocab_size = #self.index2word
  print_f(string.format("%d words and %d sentences processed in %.2f seconds.", self.total_count, n, sys.clock() - start))
  print_f(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))

  -- initalize word2vec model  
  self.w2v = self:initialize_model() 
end

function Word2Vec:initialize_model()
  -- initialize word/context embeddings now that vocab size is known
  self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
  self.context_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings
  self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
  local model = nn.Sequential()
  model:add(nn.ParallelTable())
  model.modules[1]:add(self.context_vecs)
  model.modules[1]:add(self.word_vecs)
  model:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
  model:add(nn.Sigmoid())
  self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)

  return model
end

-- Build a table of unigram frequencies from which to obtain negative samples
function Word2Vec:build_table()
  local start = sys.clock()
  local total_count_pow = 0
  print_f("Building a table of unigram frequencies... ")
  for _, count in pairs(self.vocab) do
    total_count_pow = total_count_pow + count^self.alpha
  end   
  self.table = torch.IntTensor(self.table_size)
  local word_index = 1
  local word_prob = self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
  for idx = 1, self.table_size do
    self.table[idx] = word_index
    if idx / self.table_size > word_prob then
        word_index = word_index + 1
        word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
    end
    if word_index > self.vocab_size then
        word_index = word_index - 1
    end
  end
  print_f(string.format("Done in %.2f seconds.", sys.clock() - start))
end

-- Train on word context pairs
function Word2Vec:train_pair(word, contexts)
  local p = self.w2v:forward({contexts, word})
  
  -- Backward phase is always only through word2vec model
  local loss = self.criterion:forward(p, self.labels)
  local dl_dp = self.criterion:backward(p, self.labels)
  self.w2v:zeroGradParameters()
  self.w2v:backward({contexts, word}, dl_dp)
  self.w2v:updateParameters(self.lr)
end

-- Sample negative contexts
function Word2Vec:sample_contexts(context)
  self.contexts[1] = context
  local i = 0
  while i < self.neg_samples do
    neg_context = self.table[torch.random(self.table_size)]
    if context ~= neg_context then
      self.contexts[i+2] = neg_context
      i = i + 1
    end
  end
end

-- Train on sentences that are streamed from the hard drive
-- Check train_mem function to train from memory (after pre-loading data into tensor)
function Word2Vec:train_stream(corpus)
  print_f("Training...")
  local start = sys.clock()
  local c = 0
  f = io.open(corpus, "r")
  for line in f:lines() do
    sentence = self:split(line)
    for i, word in ipairs(sentence) do
      word_idx = self.word2index[word]
      if word_idx ~= nil then -- word exists in vocab
        local reduced_window = torch.random(self.window) -- pick random window size
        self.word[1] = word_idx -- update current word
        for j = i - reduced_window, i + reduced_window do -- loop through contexts
          local context = sentence[j]
          if context ~= nil and j ~= i then -- possible context
            context_idx = self.word2index[context]
            if context_idx ~= nil then -- valid context
              self:sample_contexts(context_idx) -- update pos/neg contexts
              self:train_pair(self.word, self.contexts) -- train word context pair
              c = c + 1
              self.lr = math.max(self.min_lr, self.lr + self.decay) 
              if c % 100000 ==0 then
                  print_f(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
              end
            end
          end
        end		
      end
    end
  end
end

-- Row-normalize a matrix
function Word2Vec:normalize(m)
  m_norm = torch.zeros(m:size())
  for i = 1, m:size(1) do
    m_norm[i] = m[i] / torch.norm(m[i])
  end
  return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Word2Vec:get_sim_words(w, k)
  if type(w) == "string" then w = self:get_vector(w) end
  local sim = torch.mv(self.word_vecs_norm, w)
  sim, idx = torch.sort(-sim)
  local r = {}
  for i = 1, k do
      r[i] = {self.index2word[idx[i]], -sim[i]}
  end
  return r
end

function Word2Vec:get_vector(w)
  if self.word_vecs_norm == nil then
    self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
  end
  if type(w) == "string" then
    if self.word2index[w] == nil then
      print_f("'"..w.."' does not exist in vocabulary.")
      return nil
    else  
      w = self.word_vecs_norm[self.word2index[w]]
    end
  end

  return w
end

-- print_f similar words
function Word2Vec:print_sim_words(words, k)
  for i = 1, #words do
    r = self:get_sim_words(words[i], k)
    if r ~= nil then
      print_f("-------"..words[i].."-------")
      for j = 1, k do
        print_f(string.format("%s, %.4f", r[j][1], r[j][2]))
      end
    end
  end
end

-- split on separator
function Word2Vec:split(input, sep)
  if sep == nil then
      sep = "%s"
  end
  local t = {}; local i = 1
  for str in string.gmatch(input, "([^"..sep.."]+)") do
    t[i] = str; i = i + 1
  end
  return t
end

-- pre-load data as a torch tensor instead of streaming it. this requires a lot of memory, 
-- so if the corpus is huge you should partition into smaller sets
function Word2Vec:preload_data(corpus)
  print_f("Preloading training corpus into tensors (Warning: this takes a lot of memory)")
  local start = sys.clock()
  local c = 0
  f = io.open(corpus, "r")
  self.train_words = {}; self.train_contexts = {}
  for line in f:lines() do
    sentence = self:split(line)
    for i, word in ipairs(sentence) do
      word_idx = self.word2index[word]
      if word_idx ~= nil then -- word exists in vocab
        local reduced_window = torch.random(self.window) -- pick random window size
        self.word[1] = word_idx -- update current word
        for j = i - reduced_window, i + reduced_window do -- loop through contexts
          local context = sentence[j]
          if context ~= nil and j ~= i then -- possible context
            context_idx = self.word2index[context]
            if context_idx ~= nil then -- valid context
              c = c + 1
              self:sample_contexts(context_idx) -- update pos/neg contexts
              if self.gpu==1 then
                self.train_words[c] = self.word:clone():cuda()
                self.train_contexts[c] = self.contexts:clone():cuda()
              else
                self.train_words[c] = self.word:clone()
                self.train_contexts[c] = self.contexts:clone()
              end
            end
          end
        end	      
      end
    end
  end
  print_f(string.format("%d word-contexts processed in %.2f seconds", c, sys.clock() - start))
end

-- train from memory. this is needed to speed up GPU training
function Word2Vec:train_mem()
  local start = sys.clock()
  for i = 1, #self.train_words do
    self:train_pair(self.train_words[i], self.train_contexts[i])
    self.lr = math.max(self.min_lr, self.lr + self.decay)
    if i%100000==0 then
      print_f(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", i, sys.clock() - start, self.lr))
    end
  end    
end

-- train the model using config parameters
function Word2Vec:train_model(corpus, gpu_no)
  if self.gpu==1 then
    print_f('gpu: '.. gpu_no)
    require("cunn")
    cutorch.setDevice(gpu_no + 1)
    self:cuda()
  end
  if self.stream==1 then
    self:train_stream(corpus)
  else
    self:preload_data(corpus)
    self:train_mem()
  end
end

function Word2Vec:save_model(model_info_loc, snapshot)
  -- write model info 
  local model_info = {vocab=self.vocab, index2word=self.index2word, word2index=self.word2index, total_count=self.total_count, table=self.table}
  torch.save(model_info_loc, model_info)

  -- save model as snapshot
  torch.save(snapshot, self.w2v)
end
