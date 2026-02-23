NEXT WORD PREDICTION APP

EXPLANATION
This system predicts the next words in a sequence using a trained LSTM model based on friends1.txt file containing the dataset.

<figure>
    <img src="images/next_word predictor_app.png" alt="Predictor App Screenshot" width="600"/>
    <figcaption>A screenshot of the Predictor App's interface</figcaption>
</figure>



FEATURES
    -LSTM neural network trained on Friends dialogue stored in a dataset ('friends1.txt')
    -Simple and clean web interface
    -Option to choose from 1 to 6 words to predict
    -Real-time word predictions

HOW IT WORKS

1. The model is trained using the 'friend1.txt'
2. Text is tokenized and sequences are created to train a LSTM
3. When a phrase is entered, the model predicts the next word(s) iteratively
4. The web app displays the complete sentence

REQUIREMENTS

- Python 3.7+
- pip package manager
- TensorFlow
- Flask
- NumPy

INSTALLATION STEPS

1. Extract all files from the provided ZIP folder to your desired location

2. Open Command Prompt/Terminal and navigate to the extracted folder:
   command: cd "path_to_extracted_folder"

3. Set up a virtual environment (this is recommended but it's optional):
   in command prompt (or Linux Powershell):
   command: python -m venv venv (to create a virtualenv)
   command: path\to\venv\Scripts\activate (activate the virtual environment)
   command: cd venv\Scripts (change to Scripts directory)
   command: activate.bat (activate the virtual environment)

   Expected output: After activation, the virtual environment name appears on the left side of the terminal, indicating it's active

4. Install required Python packages:
   command: pip install tensorflow flask numpy

5. If Flask installation fails, try:
   command: pip install --user flask
   ,or
   command: python -m pip install flask

6. Train the model (this may take some time depending on your system):
   command: python WPredictor_Model.py
   Wait for "Model and tokenizer saved successfully!" message

7. Launch the web interface:
   command: python Web_App.py
8. Access the prediction interface in your web browser at:
   http://localhost:5000

GENERATED FILES AFTER TRAINING

1. next_word_model.keras - The trained neural network model file
   - Contains the learned patterns and weights
   - Required for making predictions
2. tokenizer.pkl - The text tokenizer object
   - Stores the vocabulary mapping (words to numbers)
   - Essential for processing new input text

USING THE WEB INTERFACE

1. Enter a starting phrase in the text box
   Example: "E.g I am going to"

2. Select how many words to predict (1-6) from the dropdown

3. Click "Predict Next Words" button

4. View the generated prediction below the form

TROUBLESHOOTING

If Flask installation fails:

1. Run Command Prompt as Administrator
2. Try: pip install --force-reinstall flask
3. Or: python -m pip install --upgrade flask

If you get "ModuleNotFoundError":

1. Verify all packages installed correctly with:
   pip list
2. Check Python environment matches the one you're using

If Model file not found:

- Run WPredictor_Model.py first to generate it.

Tokenizer errors:
\*Ensure tokenizer.pkl is in the same folder as Web_App.py
