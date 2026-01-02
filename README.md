# HealthRoleBench: Role-Aware Health Misinformation Benchmark

HealthRoleBench is a **synthetic, role-aware benchmark** designed to evaluate how well large language models (LLMs) can identify *who* is spreading health misinformation rather than simply determining whether content is true or false.

Unlike traditional misinformation datasets, HealthRoleBench focuses on **motivational intent**, enabling deeper analysis of how misinformation is framed, propagated, and generalized across models and diseases.

---

## 🧠 Core Motivation

Most misinformation datasets emphasize factual correctness.  
HealthRoleBench instead asks:

> *Who is speaking, and what motivates their misinformation?*

This framing enables evaluation of **intent-level generalization**, which is essential for building socially aware and robust AI systems.

---

## 🧬 Motivational Roles

| Role | Description |
|------|-------------|
| **Religious Conspiracy Theorist** | Frames disease as divine punishment or spiritual consequence |
| **Anti-Vaccine Advocate** | Opposes vaccination using pseudoscience or institutional distrust |
| **Fear Monger** | Amplifies fear, urgency, and catastrophic outcomes |
| **Misinformation Spreader** | Shares misleading information without explicit ideological intent |

---

## 🌍 Disease Domains

HealthRoleBench spans three public health contexts:

- **COVID-19**
- **HPV**
- **Influenza**

Each role is instantiated independently for each disease.

---

## 🧱 Dataset Structure

