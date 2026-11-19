# notebooks/01_eda_visuals.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean professional plot style
sns.set_theme(style="whitegrid", palette="muted")

data_path = os.path.join("artifacts", "master_cleaned_customers.csv")
df = pd.read_csv(data_path)

# Increase figure size slightly for clean spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle(
    "Olist E-Commerce Customer Churn & Behavioral EDA",
    fontsize=20,
    fontweight="bold",
    y=0.98
)

# Plot 1: Churn Class Balance (Countplot)
ax1 = sns.countplot(
    data=df, x="is_churned", hue="is_churned",
    palette="viridis", legend=False, ax=axes[0, 0]
)
axes[0, 0].set_title("1. Customer Churn Distribution (0 = Active, 1 = Churned)", fontsize=14, fontweight="bold", pad=12)
axes[0, 0].set_xlabel("Churn Status", fontsize=11)
axes[0, 0].set_ylabel("Customer Count", fontsize=11)
axes[0, 0].set_ylim(0, df["is_churned"].value_counts().max() * 1.12)  # Extra headroom so bar labels don't touch top border
for container in ax1.containers:
    ax1.bar_label(container, fmt="%d", fontsize=11, padding=4)

# Plot 2: Delivery Delay vs. Review Score Impact on Churn (Boxplot)
# Round avg_review_score to nearest whole star (1-5) so x-axis labels don't collide
plot2_df = df[df["avg_delivery_delay_days"].between(-15, 25)].copy()
plot2_df["review_star_rounded"] = plot2_df["avg_review_score"].round().astype("Int64")

sns.boxplot(
    data=plot2_df,
    x="review_star_rounded",
    y="avg_delivery_delay_days",
    hue="is_churned",
    ax=axes[0, 1],
    palette="Set2"
)
axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1.2, label="Estimated Delivery Deadline")
axes[0, 1].set_title("2. Delivery Delay (Days) vs. Review Score by Churn", fontsize=14, fontweight="bold", pad=12)
axes[0, 1].set_xlabel("Average Customer Review Score (Rounded 1–5 Stars)", fontsize=11)
axes[0, 1].set_ylabel("Delivery Delay (Days > 0 = Late)", fontsize=11)
axes[0, 1].legend(title="Churned", loc="upper right")

# Plot 3: Spend Distribution by Primary Payment Type (Violin Plot)
top_payments = df[df["primary_payment_type"].isin(["credit_card", "boleto", "voucher", "debit_card"])]
sns.violinplot(
    data=top_payments[top_payments["total_monetary_spend"] <= 500],
    x="primary_payment_type",
    y="total_monetary_spend",
    hue="is_churned",
    split=True,
    inner="quart",
    ax=axes[1, 0],
    palette="coolwarm"
)
axes[1, 0].set_title("3. Spend Distribution (≤ $500) Across Payment Methods", fontsize=14, fontweight="bold", pad=12)
axes[1, 0].set_xlabel("Primary Payment Method", fontsize=11)
axes[1, 0].set_ylabel("Total Monetary Spend (BRL)", fontsize=11)
axes[1, 0].legend(title="Churned", loc="upper right")

# Plot 4: Numerical Feature Correlation Heatmap
num_cols = [
    "is_churned", "frequency_orders", "total_monetary_spend",
    "avg_order_value", "avg_delivery_delay_days", "max_installments",
    "avg_review_score", "is_late_delivery"
]
corr_matrix = df[num_cols].corr()
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
    vmin=-1, vmax=1, linewidths=0.5, ax=axes[1, 1],
    cbar_kws={"shrink": 0.85}
)
axes[1, 1].set_title("4. Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
axes[1, 1].tick_params(axis="x", rotation=35)
axes[1, 1].tick_params(axis="y", rotation=0)

# Add generous vertical (hspace) and horizontal (wspace) spacing between subplots
plt.subplots_adjust(top=0.91, bottom=0.10, left=0.07, right=0.96, hspace=0.35, wspace=0.22)

os.makedirs("artifacts", exist_ok=True)
output_path = os.path.join("artifacts", "eda_executive_summary.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved high-resolution EDA chart to {output_path}")
plt.show()