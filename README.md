# MSCI 436 Project - Housing Price Prediction DSS

## Description:
We use a Decision Support System (DSS) and linear regression model with the MNIST House Price data set to effectively analyze large amounts of data and predict house prices. This allows us to identify significant factors affecting house prices and perform a sensitivity analysis, enhancing our understanding and providing further insights through visualized results.

## File Structure:
- `app.py` - The streamlit application code
- `model.py` - Testing and training of the LR model
- `tab1.py` - House feature input page
- `tab2.py` - Top 3 factors of the house, and sensitivity analysis
- `tab3.py` - Data visualization and heat map

Each tab is used as a separate view in the DSS. This design is so the user can focus on one task at a time, whether it be inputting data, seeing statistical insights, or viewing visualizations.

## Setup:
1. Copy the repository to your computer. run: `git clone https://github.com/twarep/436-project.git`
2. Create and activate a Python virtual environmment, you can use: `python -m venv venv`
3. To activate the virtual environment on mac/linux with all packages installed, run: `source venv/bin/activate`
4. To activate the virtual environment on mac/linux with all packages installed, run: `venv\bin\activate.bat`
5. To install requirements, run: `pip install -r requirements.txt`
6. To run the streamlit app, run: `streamlit run app.py`

### Contributors:
Group 21:
- Natalie Tam
- Kanishk Dutta
- Peter Twarecki
- Anita Yang


### Additional information:
- [MNIST Ames IA House dataset](https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv)
- [Presentation](https://docs.google.com/presentation/d/1mX_GqybdBzqAQEkJNiAC4JtFsZBsRCM8ZTGkcwnQOr0/edit?pli=1#slide=id.g2314c4ab33b_1_0)
