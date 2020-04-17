import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch

class WSJ_Dataset(Dataset):
    def __init__(self, data_name = 'train'):
        self.data_name = data_name
        if self.data_name not in ['train', 'dev', 'test']:
            print('Provide string in [train, dev, test] only')

        utterances, self.label_transcript = self.load_dataset(self.data_name)
        self.utterances = self.make_divisible(utterances)
        self.max_input_len = np.max([utt.shape[0] for utt in self.utterances])
        self.max_output_len = np.max([ls.shape[0] for ls in self.label_transcript])
    
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        return self.utterances[index], len(self.utterances[index]),\
             self.label_transcript[index], len(self.label_transcript[index])

    def load_dataset(self, name):
        utterances = np.load(f'data/{name}_new.npy', allow_pickle = True, encoding = 'latin1')
        if name == 'test':
            label_seqs = np.array([['-'] for _ in range(len(utterances))])
        else:
            label_seqs = np.load(f'data/{name}_transcripts.npy', allow_pickle = True)
            for i, sentence in enumerate(label_seqs):
                label_seqs[i] = np.array([word.decode('utf-8') for word in sentence])
        #The for loop above converts all the words in sentences from b'THE' to 'THE'
        return utterances, label_seqs
    
    def make_divisible(self, utterances):
        for i, utterance in enumerate(utterances):
            # Repeating last frame
            while len(utterances[i]) % 8 != 0:
                utterances[i] = np.concatenate((utterances[i], [utterance[-1]]), axis=0)
        return utterances

    def collate(self, batch):
        """
        Here batch is a list of tuple of __getitem__ method ie
        [(x, len(x), y, len(y)), (x, len(x), y, len(y)), ........]

        sorted_utterances_lens: sorted len(x) array
        utterances: sorted x array
        padded_utterances: contains (batch_size, max_len, 40) padded utterance squence
        """
        utterances = np.array([x[0] for x in batch])
        utterances_len = [x.shape[0] for x in utterances]
        sorted_utterances_idx = np.flipud(np.argsort(utterances_len))
        # now sort everything according to sorted_utterances_idx
        sorted_utterances_lens = np.array([x[1] for x in batch])[sorted_utterances_idx]
        utterances = utterances[sorted_utterances_idx]
        utterances_max_len = np.max(sorted_utterances_lens)
        padded_utterances = np.zeros((len(batch), utterances_max_len, 40))

        sorted_labels = np.array([x[2] for x in batch])[sorted_utterances_idx]
        sorted_label_lens = np.array([x[3] for x in batch])[sorted_utterances_idx]
        labels_max_len = np.max(sorted_label_lens)
        padded_label = np.zeros((len(batch), labels_max_len))
        label_mask = np.zeros((len(batch), labels_max_len))

        i = 0
        for utterance in utterances:
            padded_utterances[i, :len(utterance), :] = utterance
            i += 1

        i = 0
        for label in sorted_labels:
            padded_label[i, :len(label)] = label
            label_mask[i, :len(label)] = 1
            i += 1
        
        return torch.from_numpy(padded_utterances).float(), \
                torch.from_numpy(sorted_utterances_lens).int(), \
                torch.from_numpy(padded_label).long(), \
                torch.from_numpy(sorted_label_lens).int(), \
                torch.from_numpy(label_mask).long()

class WSJ_DataLoader:
    def __init__(self, args):
        self.train_dataset = WSJ_Dataset('train')
        self.val_dataset = WSJ_Dataset('dev')
        self.test_dataset = WSJ_Dataset('test')

        self.max_input_len = np.max([self.train_dataset.max_input_len] + \
                            [self.val_dataset.max_input_len] + \
                            [self.test_dataset.max_input_len])
        self.max_output_len = np.max([self.train_dataset.max_output_len] + \
                            [self.val_dataset.max_output_len])
    
        # Constructing index_to_char and char_to_index maps
        self.index_to_char, self.char_to_index = self.create_vocab_char()

        # Converting transcripts to ints
        train_label = self.convert_to_int(self.train_dataset)
        val_label = self.convert_to_int(self.val_dataset)

        # storing the int labels back into the dataset
        self.train_dataset.label_transcript = train_label
        self.val_dataset.label_transcript = val_label
        print('works till here')

        # creating dataloader here for the same, this uses: 
        """
        batch_size = args.batch_size
        pin_memory = args.cuda
        num_workers = args.num_worker
        """
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=24, \
            num_workers = 0, pin_memory = False,\
            collate_fn=self.train_dataset.collate, shuffle=True)

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=24,\
            num_workers = 0, pin_memory = False,\
            collate_fn=self.val_dataset.collate, shuffle=False)

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1,\
                                collate_fn=self.test_dataset.collate, shuffle=False)


    def create_vocab_char(self):
        index_to_char = []
        char_to_index = {}
        for i,c in enumerate(['<s>','<e>']):
            index_to_char.append(c)
            char_to_index[c] = i
        # index 0 and 1 will be used start and end character
        i = 2
        for sentence in self.train_dataset.label_transcript:
            for word in sentence:
                for c in word:
                    if c not in char_to_index:
                        char_to_index[c] = i
                        index_to_char.append(c)
                        i += 1
        for sentence in self.val_dataset.label_transcript:
            for word in sentence:
                for c in word:
                    if c not in char_to_index:
                        char_to_index[c] = i
                        index_to_char.append(c)
                        i += 1
        return index_to_char, char_to_index

    def convert_to_int(self, dataset):
        int_transcript = []
        for sentence in dataset.label_transcript:
            int_sentence = []
            int_sentence.append(0)  # start character int
            for word in sentence:
                for c in word:
                    int_sentence.append(self.char_to_index[c])
            int_sentence.append(1) # end character int
            int_transcript.append(int_sentence)
            
        return int_transcript

if __name__ == "__main__":
    print('Testing starts here: ')
    # input should be args, but its not currently defined
    DataLoaderContainer = WSJ_DataLoader([])
    for inputs in DataLoaderContainer.val_dataloader:
        print('padded_utterances shape: ', inputs[0].shape)
        print('sorted_utterances_lens shape: ', inputs[1].shape)
        print('padded_label shape: ', inputs[2].shape)
        print('sorted_label_lens: ', inputs[3].shape)
        print('label_mask: ',inputs[4].shape)
        break
