# A deep neural network for transfer learning

We aimed to extract a metamodel out of a physical model that numerically predicts the natural convection characteristics in a square enclosure, filled with a Newtonian fluid. This problem is governed by two parameters: Ra and Pr (see Appendix A of the paper for details about the mathematical analysis). We consider Ra of up to 1e8 and Pr of greater than 0.05 (1<Ra≤1e8 and 0.05≤Pr<∞); however, lower Pr were also considered provided that the ratio of Ra ⁄ Pr is at most less than 1e8.
A 400×400 grid system was shown to provide precise results for the average Nu even for the most stringent cases. Appendix B of the paper includes details related to the numerical method and grid independence test. Using a single logical processor on a 2.6 GHz Intel Core i7-3720QM CPU, an average computational time of about 4,850 seconds (as high as about 13,000 seconds for low Pr) was spent for obtaining the numerical solutions using a 400×400 grid system. Nonetheless, as demonstrated in Appendix C of the paper, lower grid systems can provide accurate numerical solutions for limited ranges of Ra. For example, a 200×200 grid system (with an average simulation time of 1,300 seconds) can reliably be used for Ra of up to 1e7 with errors of less than 0.5%. Therefore, we consider a multi-grid simulation that also uses lower grid systems, wherever possible, to decrease the simulation cost in training our AI model.
We developed a TL framework that enables us to incorporate any potential features. We started with constructing a metamodel based on one input (in this case, Ra) and later added other input parameters (namely, Pr). For simplicity, we only used 200×200 and 400×400 grid systems for our simulations. We initially generated a metamodel that predicts the variation of Nu with Ra for an air-filled enclosure (Pr=0.71). We trained an ANN using 30 data points (ranging from Ra=1 to 2e8), without using a validation dataset during the training (with the purpose of lowering the simulation cost). We then extended our metamodel to also consider the effects of Pr. We applied the same structure as Ra. We then merged the outputs of the Pr branch and block I using a Multiply layer (Keras API). We added a one-node layer after the Multiply layer to adjust for the multiplication coefficient. We froze the layers on block I, and subsequently, trained the new hidden layers using the data points from the first step as well as a new dataset of constant Ra simulation points (24 training data at a fixed Ra=1e5 and variable Pr ranging from Pr = 1e-3 to 1e5). We trained the ANN using 54 data points and validated it using 20 data points (10 data points of Pr-constant and 10 data points of Ra-constant). To increase the accuracy, one can remove the Multiply layer and replace it with a concatenated layer followed by two additional hidden layers. We froze the training on the two branches and transferred the learning from step 2. We trained the new hidden layers using new simulation points at fixed Ra=1e8 or at fixed Pr=0.05 on top of the previous dataset. Our results on the test dataset remarkably improved after this step. Further improvement in the metamodel accuracy is possible after retraining the DNN from step 3 using more data.

We continued the above procedure to build a Nu metamodel for enclosures with a centered hole as well. We considered hollow enclosures with the inner walls being adiabatic and the other boundary conditions the same as the original benchmark problem. To fully capture the variations of Nu, we satisfactorily considered dimensionless hollow diameter (d*) of sizes 0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.96, 0.97, 0.98, and 0.99. The case of d*=0 describes the original problem (with no hole at the center) and the case of d*=1 refers to the trivial case with no heat transfer domain (i.e., Nu=0). We generated a dataset of 798 new simulations for different hollow diameters and combined it with the training data from the previous section for the case of d*=0. To generate our training dataset, we used a multi-grid approach based on the values of the input Rayleigh number and hollow size. The dimensionless grid sizes for d* ranging from 0.3 to 0.9 were 1⁄160, 1⁄320, and 1⁄640 for Ra≤1e4, 1e4<Ra<1e7, and Ra≥1e7, respectively. However, for the cases with larger hollow sizes, we needed finer grid sizes as small as 1⁄1600. We have employed the already trained DNN from the benchmark problem and transferred its learning to form a metamodel for the hollow enclosure. The new structure is created by adding a new branch for variations of d* before combining the branches. Therefore, we froze the training on the two former branches (as were already trained), and trained the new branch and the last two hidden layers located after the concatenate layer. A test was carried out using a dataset of 200 simulations using the highest fidelity grid systems for d^*=0, 0.2, 0.4, 0.6, 0.78, 0.82, 0.88, 0.91, 0.94, 0.965, and 0.985. The average and maximum difference between the predicted and simulated Nu results for the TL metamodel are 0.075 and 0.803, respectively.