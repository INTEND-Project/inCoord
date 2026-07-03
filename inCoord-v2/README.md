
This repository contains the implementation used in the paper:

**Intent-Aware Multi-Domain Cloud Continuum Coordination through Hierarchical Deep Reinforcement Learning**

Follow the steps below to reproduce the experiments and results.

---

## Setup & Execution

### **1. Set up iContinuum**
Configure the iContinuum environment as described in **iContinuumAdjustments**.  
This section will be updated after publication (due to double‑blind review).  
A full deployment guide will be provided.
It is also necessary to  insert the real IP addresses in all fields marked:
[ip_removed_for_submission]


---

### **2. Train the Domain Agents**

```bash
cd ./domainAgents
python RL_training.py
```
### **3. Train the Coordinator**
```
cd ./CoordinatorTraining
python RL_relative.py
```
### **4. Run the Hierarchical Coordinator in the actual environment:**

```bash
sh ./run.sh
```
if you need to stop it, you can run:
```bash
sh ./kill.sh
```
## Hardware Setup Used in the Experiments

| Role                     | OS                 | RAM  | vCPUs |
|--------------------------|--------------------|------|-------|
| K8S Master               | Ubuntu 20.04.6 LTS | 8GB  | 4     |
| K8S Worker               | Ubuntu 20.04.6 LTS | 4GB  | 2     |
| K8S Worker               | Ubuntu 20.04.6 LTS | 4GB  | 2     |
| K8S Worker               | Ubuntu 20.04.6 LTS | 4GB  | 2     |
| Network & Monitoring VM  | Ubuntu 20.04.6 LTS | 16GB | 8     |

