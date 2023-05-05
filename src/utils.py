def question_to_dataset(questions, data, source='question', to='question'):
    clean = data
    for i, row in data.iterrows():
        # Get the qid and question for question1
        qid1 = row['qid1']
        question1 = questions.loc[questions['qid'] == qid1, source].values[0]

        # Get the qid and question for question2    
        qid2 = row['qid2']
        question2 = questions.loc[questions['qid'] == qid2, source].values[0]

        # Update the question1 and question2 columns of the dataset DataFrame
        clean.at[i, to + '1'] = question1
        clean.at[i, to + '2'] = question2

    # Remove any rows where qid1 or qid2 do not exist in qid_question_df
    clean = clean[clean['qid1'].isin(questions['qid'])]
    clean = clean[clean['qid2'].isin(questions['qid'])]
    return clean
