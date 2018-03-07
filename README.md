# Machine-Learning-by-Dr.-Brian-Ziebart
Fall 2016


### Introduction

The data is from one competetion posted on [kaggle.com](https://www.kaggle.com/wendykan/lending-club-loan-data). Lending Club is the current leading peer to peer online loan platform market share wise. Our research goal is to answer this question with a simple Y/N to a new borrower given information. Since a loan default will cause both principal and uncollected interests losses, we want to prioritize on an as high as possible default case recall rate, also consider default case F-score and overall accuracy. 

Here are some graphs descriptions. The loan status is our Y labels, and to simplify situation into binary choices, we combine the current, fully paid grace period, grace period as TRUE, “charged off, default, late” as False. The just issued are dropped. 
