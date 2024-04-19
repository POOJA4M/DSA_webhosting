from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
import sklearn 
print(sklearn.__version__)

app = Flask(__name__)

prediction_values = pd.read_csv("selected_features (1).csv")


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/cseducation')
def cseducation():
    return render_template('cseducation.html')

@app.route('/csrange')
def csrange():
    return render_template('csrange.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route("/prediction", methods = ['GET',"POST"])
def predicted():
        if request.method == 'POST':
             
             ann_income=float(request.form['Annual_Income']) 
             mon_salary=float(request.form['Monthly_Inhand_Salary'])
             amo_inv_monthly=float(request.form['Amount_invested_monthly']) 
             debt=float(request.form['Outstanding_Debt'])
             emi=float(request.form['Total_EMI_per_month'])
             interest=float(request.form['Interest_Rate'])
             delay_due=float(request.form['Delay_from_due_date'])
             inquiries=float(request.form['Num_Credit_Inquiries'])
             creditcard=float(request.form['Num_Credit_Card'])
             accounts=float(request.form['Num_Bank_Accounts'])
             creditmix=(request.form['Credit_Mix'])
             
             

             
             score_prediction =  {
                                    "Annual_Income": ann_income,
                                    "Monthly_Inhand_Salary": mon_salary,
                                    "Amount_invested_monthly": amo_inv_monthly,
                                    "Outstanding_Debt": debt,
                                    "Total_EMI_per_month": emi,
                                    "Interest_Rate": interest,
                                    "Delay_from_due_date": delay_due,
                                    "Num_Credit_Inquiries": inquiries,
                                    "Num_Credit_Card": creditcard,
                                    "Num_Bank_Accounts":accounts,
                                    "Credit_Mix":creditmix
                                    }
             
             #dataframe
             score_predictdf = pd.DataFrame([score_prediction])


             # encoding
             encoder = pickle.load(open('onehotencoder1.pkl','rb'))

             column_to_encode = ['Credit_Mix']

             
             encoded_data = encoder.transform(score_predictdf[['Credit_Mix']])
             
             encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Credit_Mix']))
           
             #scaling
             scalar = pickle.load(open('scaler1.pkl','rb'))

             numerical_features = ['Annual_Income', 'Monthly_Inhand_Salary', 'Amount_invested_monthly',
                          'Outstanding_Debt', 'Total_EMI_per_month', 'Delay_from_due_date']

             #prediction_values[numerical_features] = scalar.fit(prediction_values[numerical_features])
             scaled_data = scalar.transform(score_predictdf[numerical_features].values)
             
             scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)

             # Concatenate the scaled DataFrame and the encoded DataFrame with the original DataFrame
             transformed_df = pd.concat([scaled_df, encoded_df, score_predictdf.drop(numerical_features + column_to_encode, axis=1)], axis=1)


             

            

             #modeling
             pickled_model = pickle.load(open('rf_best.pkl','rb'))

             results = pickled_model.predict(transformed_df)  

             #Decoding result
             decoder = pickle.load(open('y_encoder1.pkl','rb'))  

             decoded_result = decoder.inverse_transform(results)   
             prediction = decoded_result.item()  

             if prediction == 'Good':
                        prediction_text = "Congratulations! Your credit score is in the 'GOOD' range, showcasing strong financial management."
             elif prediction == 'Standard':
                        prediction_text = "Your credit score falls within the 'STANDARD' range, representing average financial standing."
             elif prediction == 'Poor':
                        prediction_text = "Unfortunately, your credit score is categorized as 'POOR', suggesting challenges in obtaining favorable financial terms."
             else:
              prediction_text = "Unable to determine credit score category."
 
             

        return render_template('res.html',prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()