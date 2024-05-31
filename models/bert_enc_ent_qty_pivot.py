
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig,BertModel, BertForTokenClassification,BertModel

"""As deep learning can be accellerated a lot using a GPU instead of a CPU, make sure you can run this notebook in a GPU runtime (which Google Colab provides for free! - check "Runtime" - "Change runtime type" - and set the hardware accelerator to "GPU").

We can set the default device to GPU using the following code (if it prints "cuda", it means the GPU has been recognized):
"""

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

"""#### **Downloading and preprocessing the data**
Named entity recognition (NER) uses a specific annotation scheme, which is defined (at least for European languages) at the *word* level. An annotation scheme that is widely used is called **[IOB-tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)**, which stands for Inside-Outside-Beginning. Each tag indicates whether the corresponding word is *inside*, *outside* or at the *beginning* of a specific named entity. The reason this is used is because named entities usually comprise more than 1 word.

Let's have a look at an example. If you have a sentence like "Barack Obama was born in Hawaï", then the corresponding tags would be   [B-PERS, I-PERS, O, O, O, B-GEO]. B-PERS means that the word "Barack" is the beginning of a person, I-PERS means that the word "Obama" is inside a person, "O" means that the word "was" is outside a named entity, and so on. So one typically has as many tags as there are words in a sentence.

So if you want to train a deep learning model for NER, it requires that you have your data in this IOB format (or similar formats such as [BILOU](https://stackoverflow.com/questions/17116446/what-do-the-bilou-tags-mean-in-named-entity-recognition)). There exist many annotation tools which let you create these kind of annotations automatically (such as Spacy's [Prodigy](https://prodi.gy/), [Tagtog](https://docs.tagtog.net/) or [Doccano](https://github.com/doccano/doccano)). You can also use Spacy's [biluo_tags_from_offsets](https://spacy.io/api/goldparse#biluo_tags_from_offsets) function to convert annotations at the character level to IOB format.

Here, we will use a NER dataset from [Kaggle](https://www.kaggle.com/namanj27/ner-dataset) that is already in IOB format. One has to go to this web page, download the dataset, unzip it, and upload the csv file to this notebook. Let's print out the first few rows of this csv file:
"""

from google.colab import drive
drive.mount('/content/drive')

"""#From Here

Let's only keep the "sentence" and "word_labels" columns, and drop duplicates:
"""

# data = pd.read_csv("/content/drive/MyDrive/10krowsAttribute_corrected.csv")
data = pd.read_csv("/content/drive/MyDrive/BERT/10krowsAttribute_corrected.csv")
data.head()

df_ = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
df_.head()

df_.shape

Tag = ['O','B-ent','I-ent','B-attr','I-attr','B-qty','I-qty']
labels_to_ids = {k: v for v, k in enumerate(Tag)}
ids_to_labels = {v: k for v, k in enumerate(Tag)}
labels_to_ids,ids_to_labels

len(labels_to_ids)

# """#### **Preparing the dataset and dataloader**

# Now that our data is preprocessed, we can turn it into PyTorch tensors such that we can provide it to the model. Let's start by defining some key variables that will be used later on in the training/evaluation process:
# """

# MAX_LEN = 128
# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 1
# EPOCHS = 15
# LEARNING_RATE = 1e-05
# MAX_GRAD_NORM = 10
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

num_added_toks = tokenizer.add_tokens(["<EOS>","<SOS>"])
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
# model.resize_token_embeddings(len(tokenizer))
len(tokenizer)



################################## DATA LOADER ######################################

class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len


  def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")
        # print(sentence)
        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            #  is_pretokenized=True,
                             is_split_into_words=True,
                             return_offsets_mapping=True,
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels=[]
        label_index = 0
#         new=[1:encoding["offset_mapping"]]
        for idx,off_map in enumerate(encoding["offset_mapping"]):
#             print(off_map)
            start, end= off_map
#             print("label_index",label_index," ", (start, end))
            # If start and end offsets are both zero, append zero to encoded_labels
            if start == 0 and end == 0:
                encoded_labels.append(-100)
            elif start==0 and end!=0:
                # Otherwise, append the corresponding label to encoded_labels
                encoded_labels.append(labels[label_index])
#                 print(encoding["offset_mapping"][idx+1][])
                if encoding["offset_mapping"][idx+1][0]==0 and encoding["offset_mapping"][idx+1][1]!=0:
                    label_index += 1
                 # Move to the next label
            elif start!=0 and end!=0:
                encoded_labels.append(labels[label_index])
                if encoding["offset_mapping"][idx+1][0]==0 and encoding["offset_mapping"][idx+1][1]!=0:
                    label_index += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        item['qty_pivot'] = torch.logical_or(item['labels']==5,item['labels']==6).int()
        # item['ent_pivot'] = torch.logical_or(item['labels']==1,item['labels']==2).int()
        # sentence= self.data.sentence
        # sent={}
        # item['sent']=sentence

        return item


  def __len__(self):
        return self.len

################################## DATA LOADER ######################################

class dataset_test(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len


  def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")
        # print(sentence)
        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            #  is_pretokenized=True,
                             is_split_into_words=True,
                             return_offsets_mapping=True,
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels=[]
        label_index = 0
#         new=[1:encoding["offset_mapping"]]
        for idx,off_map in enumerate(encoding["offset_mapping"]):
#             print(off_map)
            start, end= off_map
#             print("label_index",label_index," ", (start, end))
            # If start and end offsets are both zero, append zero to encoded_labels
            if start == 0 and end == 0:
                encoded_labels.append(-100)
            elif start==0 and end!=0:
                # Otherwise, append the corresponding label to encoded_labels
                encoded_labels.append(labels[label_index])
#                 print(encoding["offset_mapping"][idx+1][])
                if encoding["offset_mapping"][idx+1][0]==0 and encoding["offset_mapping"][idx+1][1]!=0:
                    label_index += 1
                 # Move to the next label
            elif start!=0 and end!=0:
                encoded_labels.append(labels[label_index])
                if encoding["offset_mapping"][idx+1][0]==0 and encoding["offset_mapping"][idx+1][1]!=0:
                    label_index += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        item['qty_pivot'] = torch.logical_or(item['labels']==5,item['labels']==6).int()
        # item['ent_pivot'] = torch.logical_or(item['labels']==1,item['labels']==2).int()
        # sentence= self.data.sentence
        # sent={}
        item['sent']=sentence

        return item


  def __len__(self):
        return self.len

"""Now, based on the class we defined above, we can create 2 datasets, one for training and one for testing. Let's use a 80/20 split:"""

train_size = 0.98
train_dataset = df_.sample(frac=train_size,random_state=200)
test_dataset = df_.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df_.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

"""Let's have a look at the first training example:"""

# training_set[0]
ids=tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"])
pivot=training_set[0]["qty_pivot"]
labels=training_set[0]["labels"]
print(f"Ids : {ids}")
print(f"Pivot: {pivot}")
print(f"Labels: {labels}")

"""Let's verify that the input ids and corresponding targets are correct:"""

# tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"])

# for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]), training_set[0]["labels"]):
#   print('{0:10}  {1}'.format(token, label))

"""Now, let's define the corresponding PyTorch dataloaders:"""

# training_set = dataset(train_dataset, tokenizer, MAX_LEN)
# testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

"""#### **Defining the model**

Here we define the model, BertForTokenClassification, and load it with the pretrained weights of "bert-base-uncased". The only thing we need to additionally specify is the number of labels (as this will determine the architecture of the classification head).

Note that only the base layers are initialized with the pretrained weights. The token classification head of top has just randomly initialized weights, which we will train, together with the pretrained weights, using our labelled dataset. This is also printed as a warning when you run the code cell below.

Then, we move the model to the GPU.

### BERT CLASSIFIER CREATION
"""

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertEmbeddings, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertConfig, BertModel
import torch
from torch import nn

# https://discuss.huggingface.co/t/how-to-use-additional-input-features-for-ner/4364/26

class BertEmbeddingsV2(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        max_number_of_pos_tags = 2
        self.pos_tag_embeddings = nn.Embedding(max_number_of_pos_tags, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, input_ids=None, qty_pivot=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print(f"Qty Pivot: {qty_pivot}")
        pos_tag_embeddings = self.pos_tag_embeddings(qty_pivot)

        embeddings = inputs_embeds + token_type_embeddings + pos_tag_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelV2(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsV2(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        # pos_tag_ids=None,
        qty_pivot = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        """
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # print(f"Define V2 Qty_pivot: {qty_pivot.size()}")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            # pos_tag_ids=pos_tag_ids,
            qty_pivot = qty_pivot,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         print(f"Encoder first: {encoder_outputs[1:][0][1].size()}")
#         print(f"Encoder Output : {len(encoder_outputs[1:][0])}")
#         print(f"Hidden States: {len(encoder_outputs.hidden_states)}")
#         print(f"Encoder zero: {encoder_outputs[1:][0][0].size()}")
#         print(f"Hiden State 1st: {encoder_outputs.hidden_states[0][0].size()}")
#         print(f"Hiden State 2nd: {encoder_outputs.hidden_states[1][0].size()}")

        '''
        Encoder first: torch.Size([4, 128, 768])
        Encoder Output : 13      ---------- this is what we take forward
        Hidden States: 13
        Encoder zero: torch.Size([4, 128, 768])
        Hiden State 1st: torch.Size([128, 768])
        Hiden State 2nd: torch.Size([128, 768])
        '''
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# Create the BertClassfier class
class BertClassifier_1(nn.Module):
    """
    Bert Model for Classification Tasks.
    """
    def __init__(self, tokenizer, freeze_bert=False,num_labels=len(labels_to_ids),drop1=0.1,drop2=0.1,activation="relu"):

        super(BertClassifier_1, self).__init__()

        # Instantiate BERT model
        # self.embedding_1 =  nn.Embedding(n, d, max_norm=False)
        self.num_labels=num_labels
        # self.bert = BertModel.from_pretrained('bert-base-uncased',add_pooling_layer=False)
        self.activation = activation
        config = BertConfig()
        self.bertV2 = BertModelV2.from_pretrained("bert-base-uncased", config=config)
        self.bertV2.resize_token_embeddings(len(tokenizer))
        ########### BLOCKS TO BE FREEZED ##############3
        self.enc_modules = [self.bertV2.embeddings , self.bertV2.encoder.layer[:10]]
        # self.enc_modules = [self.bertV2.encoder.layer[:9]]

        # D_in, H, D_out = 768, 264, self.num_labels

        D_in, H_1,H_2, D_out = 768, 512, 264, self.num_labels

        self.fc1 = nn.Dropout(drop1)
        self.fc2 = nn.Linear(D_in,H_1)
        self.fc3 = nn.Dropout(drop2)
        self.fc5 = nn.Linear(H_1,H_2)
        self.fc6 = nn.Dropout(drop2)
        self.fc4 = nn.Linear(H_2,D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

 
        '''
        FREEZE THE LAST FEW BLOCKS
        '''
        for module in self.enc_modules:
            for param in module.parameters():
                param.requires_grad = False

#         for name, param in self.bertV2.named_parameters():
#           if name=="embeddings.pos_tag_embeddings.weight":
# #             print("Worked")
#             param.requires_grad=True

#         for name, param in self.bertV2.named_parameters():
# #           if name=="bertV2.embeddings.pos_tag_embeddings.weight":
#             print(name,param.requires_grad)
#         # self.bertV2.embeddings.pos_tag_embeddings.parameters =  True

    def forward(self, input_ids, attention_mask,qty_pivot,labels):
      ## ADD MASK LABELS as well here, positioning or something,
      # def forward(self, encoding):

        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        outputs = self.bertV2(input_ids=input_ids,attention_mask=attention_mask,qty_pivot=qty_pivot)
        # print("Last Hidden State size: ", outputs['last_hidden_state'].size())
        ## find the output where the pivoting for masking is done from the dataset, send that to get the MASK prediction into one of the labels
        self.labels=labels

        if self.activation=="relu":
            x = self.fc2(outputs.last_hidden_state)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.fc5(x)
            x = self.relu(x)
            x = self.fc6(x)

            logits = self.fc4(x)


        else:
            x = self.fc1(outputs.last_hidden_state)
            x = self.fc2(x)
            x = self.tanh(x)
            x = self.fc3(x)
            logits = self.fc4(x)


        return outputs,logits

# model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
# model.to(device)

# model = BertClassifier_1(tokenizer=tokenizer,freeze_bert=True)
# model.to(device)

"""# Utilities"""

# Defining the training function on the 80% of the dataset for tuning the bert model
def train(MAX_GRAD_NORM,training_loader):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training model
    model.train()

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)
        qty_pivot = batch['qty_pivot'].to(device, dtype = torch.long)
        # ent_pivot = batch['ent_pivot'].to(device, dtype = torch.long)


        outputs = model(input_ids=ids, attention_mask=mask,qty_pivot=qty_pivot,labels=labels)
        # initial_loss,tr_logits,hidden_states = outputs[0],outputs[1],outputs[2]
        hidden_states,tr_logits = outputs[0],outputs[1]

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        initial_loss = loss_fn(tr_logits.view(-1, 7), labels.view(-1))
        tr_loss += initial_loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps

        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)

        # print("Active logits",active_logits.shape)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        initial_loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


######################################### VALIDATION FUNCTION GIVING THE F1 score #########################3

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            qty_pivot = batch['qty_pivot'].to(device, dtype = torch.long)
            # ent_pivot = batch['ent_pivot'].to(device, dtype = torch.long)
            with torch.no_grad():
                outputs = model(input_ids=ids, attention_mask=mask,qty_pivot=qty_pivot,labels=labels)

            hidden_states,eval_logits = outputs[0],outputs[1]
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            initial_loss = loss_fn(eval_logits.view(-1, 7), labels.view(-1))

            eval_loss += initial_loss.item()
            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # print("labels",labels)
            # print("Falttened",flattened_predictions)
            # print("active logits",eval_logits.shape)
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            # print(labels.shape)
            eval_labels.append(labels)
            eval_preds.append(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    # labels = [ids_to_labels[id.item()] for id in eval_labels]
    # predictions = [ids_to_labels[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions,eval_labels,eval_preds


##################################### CONVERT PREDICTIONS TO TAGS ############################

def convert_tensor_to_tags(tensor_):
  tags = []
  for ner_id in tensor_:
      if int(ner_id) in ids_to_labels.keys():
          tags.append(ids_to_labels[int(ner_id)])

  return tags

############################### F1 SCORE CALCULATION ########################################

def f1_score_calculation(eval_preds,eval_labels):
  # convert_tensor_to_tags(eval_preds)
  pred_list,ref_list = [],[]
  for pred,ref in zip(eval_preds,eval_labels):
    pred_list.append(convert_tensor_to_tags(pred))
    ref_list.append(convert_tensor_to_tags(ref))
    # print("Preed and Ref",pred,ref)

  labels = ['O','B-ent','I-ent','B-attr','I-attr','B-qty','I-qty']
  # Initialize lists to store F1 scores for each label
  label_f1_scores = []
  # print(labels)
  # Loop through the unique labels
  for label in labels:
      true_labels,pred_labels=[],[]
      for ref in ref_list:
        for vals in ref:
          # print(vals)
          if vals==label:
            true_labels.append(1)
          else:
            true_labels.append(0)

      for pred in pred_list:
        for vals in pred:
          # print(vals)
          if vals==label:
            pred_labels.append(1)
          else:
            pred_labels.append(0)

      f1 = f1_score(true_labels, pred_labels)

      # Append the F1 score to the list
      label_f1_scores.append(f1)

  print(label_f1_scores)


"""# Training"""

import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

parameters = {
    "drop1":[0.1],
    "MAX_GRAD_NORM": [3],
    "drop2" : [0.1],
    "LEARNING_RATE": [1e-5],
    "activation": ["relu"]
}
EPOCHS = 50
for comb in itertools.product(parameters["drop1"],
                              parameters["MAX_GRAD_NORM"],
                              parameters["drop2"],
                              parameters["LEARNING_RATE"],
                              parameters["activation"]):
    print("Current Combination: ",comb)
    f1_score_list = []
    drop1,MAX_GRAD_NORM,drop2,LEARNING_RATE,activation = comb[0],comb[1],comb[2],comb[3],comb[4]


    # MAX_LEN, MAX_GRAD_NORM, TRAIN_BATCH_SIZE, LEARNING_RATE
    TRAIN_BATCH_SIZE=8
    train_size = 0.99
    train_dataset = df_.sample(frac=train_size,random_state=200)
    test_dataset = df_.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # print("FULL Dataset: {}".format(df_.shape))
    # print("TRAIN Dataset: {}".format(train_dataset.shape))
    # print("TEST Dataset: {}".format(test_dataset.shape))
    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    ######################## WE CAN DO GRID SEARCH CV HERE ##########3##########
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    ###################### INITIALIZING THE MODEL FOR EACH SPLIT #####################################
    # model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model = BertClassifier_1(tokenizer=tokenizer,freeze_bert=True,drop1=drop1,drop2=drop2,activation=activation)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"Training epoch: {epoch + 1}")
        train(MAX_GRAD_NORM,training_loader)
        end_time = time.time()
        time_difference_seconds = end_time - start_time
        time_difference = timedelta(seconds=time_difference_seconds)
        # Extract minutes and seconds
        minutes = time_difference.seconds // 60
        seconds = time_difference.seconds % 60
        # Print time difference for the epoch
        print(f"Epoch {epoch + 1}: Time taken - {minutes} minutes {seconds} seconds")

    print("\n############### COMBINATION DONE #################")
    # for epoch in range(EPOCHS):
    #   print(f"Training epoch: {epoch + 1}")
    #   train(epoch,MAX_GRAD_NORM)

    labels, predictions,eval_labels,eval_preds = valid(model, testing_loader)

    f1_score_calculation(eval_preds,eval_labels)

labels, predictions,eval_labels,eval_preds = valid(model, testing_loader)

f1_score_calculation(eval_preds,eval_labels)

torch.save(model,"/content/drive/MyDrive/10k_BERT_EmbUnTr_10Enc_frz")
