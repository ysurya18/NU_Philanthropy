# Donor-Propensity-Modeling-Strategic-Segmentation
## The Challenge
Northeastern University's Advancement team faced a critical problem: identifying which donors were most likely to make planned gifts of $100,000 or more. With only 0.1% of their 318,000+ constituents having established planned gifts, the existing approach—relying on officer intuition and manual review—wasn't scalable. They needed a data-driven framework to prioritize cultivation efforts and focus limited resources on the highest-potential prospects.
## Our Approach
Working in a 4-person team, I led the predictive modeling and feature engineering efforts to solve this challenge:
### Data Challenges & Solutions:
* Addressed 53.9% missing age data by engineering behavioral proxies (giving recency, cumulative giving, engagement scores) that captured donor maturity without demographic imputation
* Merged and standardized multiple flat files, creating constituent-level summary features from giving history, event attendance, and volunteer engagement data
* Built three distinct feature sets optimized for different algorithms, balancing model interpretability with predictive power
### Modeling & Evaluation:
* Developed and compared three predictive models: Logistic Regression (for interpretability), XGBoost, and LightGBM
* Used precision-recall AUC and top-K capture rates instead of traditional accuracy metrics due to extreme class imbalance (0.1% positive class)
* Selected XGBoost as the operational model despite LightGBM's higher overall PR-AUC because it concentrated 2.5x more planned gift donors in the critical top 1% tier where Advancement invests most heavily
### Feature Engineering Insights:
* Created interaction terms combining capacity, engagement, and relationship signals (e.g., spouse-alumni connections, US resident × engagement)
* Engineered quadratic features to capture non-linear wealth effects—donors with $500K+ cumulative giving showed exponentially higher planned gift likelihood
* Identified that consistent giving patterns were far more predictive than isolated large gifts, challenging conventional fundraising assumptions
## Key Results
### Model Performance:
* 37.7% capture rate in top 1% tier (3,182 donors)—gift officers could focus on just 1% of constituents and reach nearly 40% of all likely planned gift prospects
* 67% capture rate in top 5% tier (15,910 donors)—maintained strong performance across mid-level cultivation pipeline
* 377x improvement in efficiency compared to random selection at the top tier
### Business Impact:
* Reduced outreach volume by 90% while capturing 75% of likely prospects through targeted top-10% strategy
* Created actionable three-tier segmentation framework with cultivation strategies tailored to each likelihood level
* Delivered CRM integration roadmap with quarterly score updates to maintain dynamic prospect identification
### Analytical Insights:
* Spouse-alumni connections emerged as strongest predictor (0.43 coefficient), revealing that dual-household ties dramatically increase planned giving propensity
* Geographic proximity amplifies engagement effects—US residents with high engagement showed elevated likelihood, enabling focused face-to-face cultivation
* Multiple giving entry points matter more than campaign participation—donors supporting diverse designations are significantly more likely prospects
## Strategic Recommendations
We translated our model insights into operational recommendations for University Advancement:
1. Portfolio Restructuring: Redesign gift officer portfolios around predicted likelihood tiers rather than geography or alphabetical assignment
2. Tier-Specific Playbooks: High-touch cultivation (1:1 estate planning meetings) for Tier 1, semi-personalized programming for Tier 2, scalable digital campaigns for Tier 3
3. Personalized Messaging: Target spouse-alumni households with legacy-focused messaging; approach high lifetime givers with estate planning conversations
4. Dynamic Monitoring: Implement alerts for behavior changes (giving acceleration, new volunteering) to identify "risers" moving between tiers
## What I Learned
This project reinforced the importance of understanding business context when selecting models. While LightGBM had better global performance (PR-AUC of 0.327 vs. 0.051), XGBoost delivered superior results where it actually mattered—the top 1-5% of prospects where Advancement concentrates cultivation resources. The "best" model on paper isn't always the best model for the business problem.
I also gained experience navigating messy real-world data. Rather than forcing demographic imputation that would introduce bias, we identified correlated behavioral signals that captured the same underlying patterns. This principled approach to missing data handling was crucial to model reliability.
## Tools & Technologies: Python, XGBoost, LightGBM, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
