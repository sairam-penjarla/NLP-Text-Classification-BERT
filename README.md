# NLP-Text-Classification-BERT

This repository contains code for text classification using BERT. The model is trained and evaluated using the `simpletransformers` library. 

## Repository Contents

- `main.py`: Main script to load data, train, evaluate, and save the text classification model.
- `requirements.txt`: List of required packages.
- `Roughbook.ipynb`: Jupyter notebook used during the development of the code.

## Usage

clone the repository:
```bash
git clone https://github.com/sairam-penjarla/NLP-Text-Classification-BERT.git
```

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Running the Script

To run the text classification script, use:

```bash
python main.py
```

## Important Notes:

* Replace `'Shared-Task-2019_Data_germeval2019.training_subtask1_2.txt'` and `'Shared-Task-2019_Data_germeval2019.training_subtask3.txt'` with the actual paths to your CSV files in the `load_data` method.
* This code uses the `simpletransformers` library for BERT-based text classification. Ensure you have this library installed and configured properly.
* For detailed information on the BERT model you're using (`distilbert-base-german-cased` in this case), consult its documentation for specific requirements and usage instructions.

## Further Considerations:

* Consider adding unit tests to ensure the correctness of the code.
* Explore advanced classification techniques using different BERT models or fine-tuning for specific use cases.
* Provide example usage for various input formats (e.g., text files, APIs).
* Include documentation or comments within the code to enhance readability and understanding.

## Contributing

We welcome contributions to this project! Feel free to submit pull requests for improvements, bug fixes, or new features. Please follow these guidelines:

* Fork the repository.
* Create a new branch for your changes.
* Implement your modifications and add unit tests if applicable.
* Submit a pull request with a clear description of your changes.
