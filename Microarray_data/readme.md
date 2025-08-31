# Microarrays

Microarrays are a collection of DNA probes that are usually bound in defined positions to a solid surface, such as a glass slide, to which sample DNA fragments can be hybridised. The probes are generally oligonucleotides that are *ink-jet printed* onto slides (Agilent) or synthesised **in situ** (Affymetrix). Labelled single-stranded DNA or antisense RNA fragments from a sample of interest are hybridised to the DNA microarray under high stringency conditions. The amount of hybridisation detected for a specific probe is proportional to the number of nucleic acid fragments in the sample.

---

## One-colour or Two-colour Arrays?

A major design consideration in a microarray experiment is whether to measure the expression levels from each sample on separate microarrays (**one-colour array**) or to compare relative expression levels between a pair of samples on a single microarray (**two-colour array**) (Figure 2). The overall performance of one-colour and two-colour arrays is similar.
<img width="972" height="752" alt="image" src="https://github.com/user-attachments/assets/e6230b3a-6cd2-47fa-bb95-f6ca8c1c59d0" />

- **Two-colour microarrays**:  
  Two biological samples (experimental/test sample and control/reference sample) are labelled with different fluorescent dyes, usually **Cyanine 3 (Cy3)** and **Cyanine 5 (Cy5)**. Equal amounts of labelled cDNA are then simultaneously hybridised to the same microarray chip. After this competitive hybridisation, the fluorescence measurements are made separately for each dye and represent the abundance of each gene in one sample (test sample, Cy5) relative to the other one (control sample, Cy3).  

  The hybridisation data are reported as a **ratio of the Cy5/Cy3 fluorescent signals** at each probe.

- **One-colour microarrays**:  
  Each sample is labelled and hybridised to a separate microarray, and we get an **absolute value of fluorescence** for each probe.

---

## Limitations of Microarrays

Hybridisation-based approaches are high throughput and relatively inexpensive, but have several limitations which include:

- reliance upon existing knowledge about the genome sequence  
- high background levels owing to cross-hybridisation  
- limited dynamic range of detection owing to both background and saturation signals  
- comparing expression levels across different experiments is often difficult and can require complicated normalisation methods  

---

## Conclusion

Microarrays have been the standard for high-throughput gene expression studies before the introduction of **RNA sequencing (RNA-seq)** in the first half of the 2010s. While RNA-seq has replaced microarrays for most transcriptional profiling studies, DNA chips are still in use. An example is the **Single Nucleotide Polymorphism (SNP) array** used in diagnostics.
# Counteracting Dye Bias Effects When Using Two-colour Arrays

One issue for two-colour arrays is related to **dye bias effects** introduced by the slightly different chemistry of the two dyes. It is important to control for this dye bias in the design of your experiment, for example by using a **dye swap design** or a **reference design** (Figure 3).

- **Dye swap design**:  
  The same pairs of samples (test and control) are compared twice with the dye assignment reversed in the second hybridisation.  

- **Reference design**:  
  The most common design for two-colour microarrays. Each experimental sample is hybridised against a common reference sample.  

---

<img width="982" height="481" alt="image" src="https://github.com/user-attachments/assets/a11619ac-6ff9-45e9-a371-41c2c4d35631" />


- **Dye Swap Design**:  
  Involves hybridising the same two samples (test and control) twice.  
  - In the first hybridisation, the control sample is labelled with **Cy3 dye** (cyan) and the test sample with **Cy5 dye** (red).  
  - In the second hybridisation, dye assignment is reversed.  

- **Reference Design**:  
  Both test and control samples are labelled with one dye (usually **Cy5**), while a reference sample is labelled with the other dye (usually **Cy3**) and co-hybridised.

---

## Performing Replicates

Replicates are essential for reliably detecting **differentially expressed genes** in microarray experiments.  

- Without replicates, no statistical analysis of the significance and reliability of the observed changes is possible.  
- The typical result of omitting replicates is an increased number of both **false-positive** and **false-negative** errors.  

### Types of Replicates
- **Technical replicates** – repeated measurements of the same biological sample.  
- **Biological replicates** – independent biological samples, representing natural biological variation.  

In **Expression Atlas**, a minimum acceptable number of biological sample replicates (**three**) is enforced to ensure sufficient statistical power to detect differential expression.
