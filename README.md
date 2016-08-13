Text analytics to automatically review medical publications in PubMed

The dataset clinical_train.csv comes from PubMed search and contains the following
variables:

title: title of the publication

abstract: abstract of the publication

trial: search result label with whether the paper is a clinical trial testing
a drug therapy for cancer (1) and (0) if the paper is not a clinical trial.

To prepare the data for text analytics, we convert titles and abstracts to text
corpora. We then convert the words to lowercase, remove the punctuation, remove
the English common words, and stem the words.

We then limit the sparseness of 95% which keeps the words that appear in at
least 5% of documents.



Regression Tree Model to classify clinical trials

We split the prepared data 70/30 into train and test datasets. Our baseline
model predicts the most frequent outcome in the training dataset - trail = 0
(no clinical trail testing a drug therapy for cancer). The baseline model
accuracy is 0.56.

We build a regression tree model (CART) to predict the probability that the
search result is a clinical trial (trail = 1). We plot the resulting tree to
determine variables the model split on. Assuming a probability threshold
of 0.5, the training set accuracy of the model is 0.82, sensitivity is 0.77 and
specificity is 0.86. The testing set accuracy is 0.76 and AUC (area under ROC
curve) is 0.837.

Decision-maker tradeoffs
We can use a CART model to predict which papers are clinical trials (SET A) and
predicted papers not to be trials (SET B). Then a researcher can manually
review SET A papers and perform the study-specific analysis.

The cost of the model making a false negative prediction is a paper that should
have been included in SET A but was missed by the model which will affect the
quality of the analysis.

The cost of the model making a false positive prediction is a paper that will be
included in SET A by mistake, adding extra work for a manual review but will
not affect the quality of the analysis.

Based on this analysis, a false negative is more costly than a false positive,
and we recommend using a probability threshold of <0.5.
