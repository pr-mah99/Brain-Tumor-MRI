import matplotlib.pyplot as plt

def plot_class_distribution(class_counts, split_name="Validation", save_path=None):
    """
    class_counts: dict بصيغة {'glioma': 607, 'meningioma': 441, 'notumor': 387, 'pituitary': 495}
    split_name: اسم المجموعة (Training / Validation / Test) ليظهر في العنوان
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = ['#F08080', '#7FDBC8', '#4FB8D8', '#A8D5BA']  # نفس ألوان الصورة الأصلية

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)

    # إضافة القيم فوق كل عمود
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )

    ax.set_title(f"Class Distribution ({split_name} Set)", fontsize=20, fontweight='bold')
    ax.set_xlabel("Classes", fontsize=16)
    ax.set_ylabel("Number of Samples", fontsize=16)
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ===== الأرقام الفعلية من مجموعتك =====
train_counts = {'glioma': 3018, 'meningioma': 2183, 'notumor': 1945, 'pituitary': 2504}   # 9650 كلي (قبل تقسيم val)
val_counts   = {'glioma': 607,  'meningioma': 441,  'notumor': 387,  'pituitary': 495}
test_counts  = {'glioma': 755,  'meningioma': 546,  'notumor': 487,  'pituitary': 626}

plot_class_distribution(train_counts, split_name="Training",   save_path="train_class_distribution.png")
plot_class_distribution(val_counts,   split_name="Validation", save_path="val_class_distribution.png")
plot_class_distribution(test_counts,  split_name="Test",       save_path="test_class_distribution.png")