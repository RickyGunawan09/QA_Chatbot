# QA_Chatbot
QA Chat bot using langchain and google gemini
![image](https://github.com/RickyGunawan09/QA_Chatbot/blob/main/img/screenshoot.png)

how to run this project:
1. download spotify data review from this [link data](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews/) and place it in data folder
2. install the requirement for this project with command `pip install -r requirements.txt`
3. run preprocess_fix to get chroma persist with command `python preprocess_fix.py`
4. please insert the google api key at function_call.py (if you didn't have one you can get it from this web [google ai studio](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwivqLys9NGEAxWi7RYFHSFDDtIYABAAGgJ0bA&ase=2&gclid=Cj0KCQiA84CvBhCaARIsAMkAvkKQ1Gb6OzQcjNp8GX5wTliK1TDVQRKgOIyufriYT1SaN2r7WHeCJPwaAqfsEALw_wcB&ei=4C7hZbGsLuXZseMPyNe54AE&ohost=www.google.com&cid=CAESVeD2G-CXtZDNpevClq6JhgBsORrNAsmWv-64N6hWsCGpZ_mllOH6dnFOkizzVB7bG5_eLFkXGN5U9R63Uerkb1u_IS7NipY7vdcmY-lkt564UNWioww&sig=AOD64_12qa_1p8jTsCvaTjLBeYCbDbwL8Q&q&sqi=2&nis=4&adurl&ved=2ahUKEwixnLSs9NGEAxXlbGwGHchrDhwQ0Qx6BAgMEAE))
5. after that run main.py with streamlit `streamlit run main.py`
