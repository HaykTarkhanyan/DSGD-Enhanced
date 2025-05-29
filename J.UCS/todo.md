# Todo revision, 2nd iter
## 
[x] text describing how reviewers comments were addressed
[x] merge 1 and 2.1. Start 2.1 with "Below we outline..."
[x] Second red paragraph -> "In Section 2 we specify the concept of representativeness and its estimation, as well as the rule confidence computation, and the enhanced MAF initialization. Section 3 describes evaluation results for our approaches, while Section 4 concludes the paper with notes on the future work."
[x] Section 2 title -> Principles behind DSGD++
[x] remove all subsection-s, use bold/italic for emphasis
[x] start section 2 with "Let's start introducing algorithms for..."
[x] section 3 title -> Evaluation
[x] section 3 - remove subsections (except 3.2)
[x] section 3.2 title -> Uncertainty Analysis and a New Approach to Rule Importance Estimation
[x] remove subsections
[] cite CNN paper


## Reviewer 1
[x] Improve the connectivity between sections for easier reading.
[] Discussion section is missing. - Future work kind of covers this
[x] The description of the results of experiments should be improved. The sections are too short for clarity.
[x] Reference section should be improved.

Minor issues:
[] Text needs polishing - very slight changes
[] The design of tables should be improved - only changed one table

## Reviewer 2 (very good reviewer)
[x] The work is not hard to follow but the paper would benefit with a brief explanation of how the MAFs are used to predict a class for people that are not familiar with the DSGD classifier.
[x] Overall the paper results are very clear, thanks for providing the source code for the classifier.
[x] Future work could be extended. It would be interesting to compare the initialization of MAFs using clustering versus expert knowledge. Do the authors have any experience with this? Have the authors tried or expect to try this improvement in bigger datasets?Just a comment on these topics with the authors experience or thoughts would be welcomed. 

# Text
We are very thankful for the reviewer's comments and suggestions. Here is how we addressed them:
1. Reduced the number of sections and made transitions between them smoother
2. Expanded the discussion of experimental results by explicitly mentioning their effects on the epoch count.
3. Added a few sentences on how MAFs are used for prediction
4. Extended the Future Work section to highlight potential integration of rule mining algorithms and emphasized the necessity of evaluating the approach on large datasets. Additionally, we acknowledged that our initialization technique was not compared with  expert knowledge-based initialization, recognizing this is an interesting direction to explore in the future.

Minor changes:
1. Improved the design of Table 1
2. Polished the text and improved the references section.

# ToDo revision, 1st iter

[x] Change refence style
[] Go over the JUCS new requirements list (sent in an email)
[x] Cite our work
[x] Integrate the appendix into text
[] Add connecting points for each section
[x] “to capture various numeric interactions among features”. The notion of features remains vague, please explain in more detail.
[] “each cluster’s ”color” is treated as a label” li, i corresponds to a color?
[x] A few lines of commentary in the two algorithms would be helpful, cf. line 13, Algorithm 2
[x] Algorithm 1 Please check Z-score exceeds zScoreThrehold
[x] DBSCAN [Ester et al. 1996] algorithm . see https://cdn.aaai.org/KDD/1996/KDD96-037.pdf This algorithm has three entries DBSCAN (SetOfPoints, Eps, MinPts) - Please explain: Fit DBSCAN on X, meaning of MaxEps
[x] Minor issues: Additional commas and periods necessary: ..., but | Here, …. than others, ….
[x] Page 1: [Valdivia et al. 2024]..-> [Valdivia, 24].
[] Algorithm 2, , which (check this after compile)
[x]Mass assignment function (MAF) several times defined MAF not used

DBSCAN [Ester et al. 1996] algorithm . see https://cdn.aaai.org/KDD/1996/KDD96-037.pdf This algorithm has three entries DBSCAN (SetOfPoints, Eps, MinPts) - Please explain: Fit DBSCAN on X, meaning of MaxEps

Page 1: [Valdivia et al. 2024]..-> [Valdivia, 24].

Page 2:
Algorithm 2, , which
Mass assignment function (MAF) several times defined MAF not used
