**Codebase behind papers:** 

[1] DSGD++: Reducing Uncertainty and Training Time in the DSGD Classifier through a Mass Assignment Function Initialization Technique, in review at J.UCS 
https://drive.google.com/file/d/1doJSTP_MiDnNVe34oQDQm4ILofgEh6Zh/view?usp=sharing

_Abstract_ 

Several studies have shown that the Dempster–Shafer theory (DST) can be successfully
applied to scenarios where model interpretability is essential. Although DST-based algorithms
offer significant benefits, they face challenges in terms of efficiency. We present a method for the
Dempster-Shafer Gradient Descent (DSGD) algorithm that significantly reduces training time—by
a factor of 1.6—and also reduces the uncertainty of each rule (a condition on features leading to a
class label) by a factor of 2.1, whilepreserving accuracycomparabletootherstatistical classification
techniques. Our main contribution is the introduction of a ”confidence” level for each rule. Initially,
wedefine the ”representativeness” of a data point as the distance from its class’s center. Afterward,
each rule’s confidence is calculated based on representativeness of data points it covers. This
confidence is incorporated into the initialization of the corresponding Mass Assignment Function
(MAF), providing a better starting point for the DSGD’s optimizer and enabling faster, more effec
tive convergence. The code is available at https://github.com/HaykTarkhanyan/DSGD-Enhanced.


[2] Improving the DSGD Classifier with an Initialization Technique for Mass Assignment Functions Tarkhanyan, A. and Harutyunyan, A Codassca 2024 137-142 Logos 2024 
https://www.logos-verlag.de/ebooks/OA/978-3-8325-5855-0.pdf


**Based on the work by Peñafiel et al.** https://github.com/Sergio-P/DSGD

Repo was previously here -> https://github.com/HaykTarkhanyan/CDSGD, 
Moved on May 1, 2024

Repo in a mess, sorry for that. :-) Feel free to ask questions (i. e. by opening an issue) if you have any
