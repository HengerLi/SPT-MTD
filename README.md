# SPT-MTD
Moving target defense has emerged as a critical paradigm of protecting a vulnerable system against persistent and stealthy attacks. To protect a system,  a defender proactively changes the system configurations to limit the exposure of security vulnerabilities to potential attackers. In doing so, the defender creates asymmetric uncertainty and complexity for the attackers, making it much harder for them to compromise the system. In practice, the defender incurs a switching cost for each migration of the system configurations. The switching cost usually depends on both the current configuration and the following configuration. Besides,  different system configurations typically require a different amount of time for an attacker to exploit and attack. Therefore, a defender must simultaneously decide both the optimal sequence of system configurations and the optimal timing for switching. In this paper, we propose a Markov Stackelberg Game framework to precisely characterize the defender's spatial and temporal decision-making in the face of advanced attackers. We introduce a value iteration algorithm that computes the defender's optimal moving target defense strategies. Empirical evaluation on real-world problems demonstrates the advantages of the Markov Stackelberg game model for spatial-temporal moving target defense.

Code is for our AAMASS 2020 paper: http://www.cs.tulane.edu/~zzheng3/publication/AAMAS20_SPTMTD.pdf


## Citation
If you find our work useful in your research, please consider citing:
```
@article{li2020spatial,
  title={Spatial-temporal moving target defense: A markov stackelberg game model},
  author={Li, Henger and Shen, Wen and Zheng, Zizhan},
  journal={arXiv preprint arXiv:2002.10390},
  year={2020}
}
