import numpy as np 
from torch.utils.data import Dataset, DataLoader

class WSJ_DataLoader(DataLoader):
    def __init__(self, args):
        self.train_dataset = WSJ_Dataset('train')
        self.val_dataset = WSJ_Dataset('dev')
        self.test_dataset = WSJ_Dataset('test')

        self.max_input_len = np.max([self.train_dataset.max_input_len] + \
                            [self.val_dataset.max_input_len] + \
                            [self.test_dataset.max_input_len])
        self.max_output_len = np.max([self.train_dataset.max_output_len] + \
                            [self.val_dataset.max_output_len])
        
        print('completed till here')




class WSJ_Dataset(Dataset):
    def __init__(self, data_name = 'train'):
        self.data_name = data_name
        if self.data_name not in ['train', 'dev', 'test']:
            print('Provide string in [train, dev, test] only')

        self.utterances, self.label_transcript = load_dataset(self.data_name)
        self.max_input_len = np.max([utt.shape[0] for utt in self.utterances])
        self.max_output_len = np.max([ls.shape[0] for ls in self.label_transcript])
    
    def __getitem__(self, index):
        """
        For single index this will return:
        self.utterances[index]: (seq_len_input, mel_bins = 40)
        len(self.utterances[index]): seq_len_input
        self.label_seqs[index]: (seq_len_output, -)
        len(self.label_seqs[index]): seq_len_output
        """
        return self.utterances[index], len(self.utterances[index]), \
            self.label_transcript[index], len(self.label_transcript[index])

    def __len__(self):
        return len(self.utterances)

def load_dataset(name):
    utterances = np.load(f'data/{name}_new.npy', allow_pickle = True, encoding = 'latin1')
    if name == 'test':
        label_seqs = np.array([['-'] for _ in range(len(utterances))])
    else:
        label_seqs = np.load(f'data/{name}_transcripts.npy', allow_pickle = True)
        for i, sentence in enumerate(label_seqs):
            label_seqs[i] = np.array([word.decode('utf-8') for word in sentence])
    #The for loop above converts all the words in sentences from b'THE' to 'THE'
    return utterances, label_seqs

def create_vocab_char(train_dataset, dev_dataset):
    vocab = []
    charset = {}
    for i,c in enumerate(['<s>','<e>']):
        vocab.append(c)
        charset[c] = i
    # index 0 and 1 will be used start and end character
    i = 2
    for sentence in train_dataset.label_transcript:
        for word in sentence:
            for c in word:
                if c not in charset:
                    charset[c] = i
                    vocab.append(c)
                    i += 1
    for sentence in dev_dataset.label_transcript:
        for word in sentence:
            for c in word:
                if c not in charset:
                    charset[c] = i
                    vocab.append(c)
                    i += 1
    return vocab, charset

def convert_to_int(train_dataset, charset):
    int_transcript = []
    for sentence in train_dataset.label_transcript:
        int_sentence = []
        int_sentence.append(0)  # start character int
        for word in sentence:
            for c in word:
                int_sentence.append(charset[c])
        int_sentence.append(1) # end character int
        int_transcript.append(int_sentence)
        
    return int_transcript

if __name__ == "__main__":
    print('Testing starts here: ')
    dev_dataset = WSJ_Dataset('dev')
    # train_dataset = WSJ_Dataset('train')
    x, len_x, y, len_y = dev_dataset.__getitem__(0)
    print('utterances: ')
    # collate_lines([np.random.rand(3, i, 40) for i in range(1,5)])
    # print('collate: ')
    vocab, charset = create_vocab_char(dev_dataset, dev_dataset)
    print(len(vocab))
    int_transcript = convert_to_int(dev_dataset, charset)
    print(len(int_transcript))