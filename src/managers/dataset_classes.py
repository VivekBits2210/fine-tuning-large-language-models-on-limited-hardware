from torch.utils.data import Dataset
class MultilabelDataset(Dataset):
    def __init__(self, pandas_df, tokenizer, max_length=1024):
        self.data = pandas_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Join title and abstract as specified
        text = f"{row['TITLE']}: {row['ABSTRACT']}"

        # Tokenize the combined text without truncation
        full_inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=False
        )

        # Tokenize with possible truncation
        truncated_inputs = self.tokenizer.encode_plus(
            text,
            None,
                add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        # Check if text was truncated
        if len(full_inputs['input_ids']) > self.max_length:
            truncated_tokens = full_inputs['input_ids'][self.max_length:]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            print(f"Text at index {index} was truncated. Truncated text: {truncated_text}")

        # Extract the one-hot encoded labels for the given row
        labels = row[self.label_cols].values.astype(int).tolist()

        return {
            'input_ids': torch.tensor(truncated_inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(truncated_inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(truncated_inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)  # Convert labels to tensor here
        }
