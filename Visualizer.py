import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Visualizer():
    def __init__(self):
        super().__init__()

    def plotHotMap(self, peptide_feature, chrom_feature, label, fig_name):
        peptide_t = peptide_feature[label == 1]
        peptide_f = peptide_feature[label == 0]
        chrom_t = chrom_feature[label == 1]
        chrom_f = chrom_feature[label == 0]
        sim_t = F.cosine_similarity(chrom_t[:, None, :], peptide_t[None, :, :], dim=-1).cpu().detach().numpy()
        sim_f = F.cosine_similarity(chrom_f[:, None, :], peptide_f[None, :, :], dim=-1).cpu().detach().numpy()
        dist_t = torch.cdist(chrom_t, peptide_t, p=2).cpu().detach().numpy()
        dist_f = torch.cdist(chrom_f, peptide_f, p=2).cpu().detach().numpy()
        plt.figure(figsize=(10, 6))  # 设置图形的大小
        plt.subplot(1, 2, 1)
        plt.imshow(sim_t, cmap='hot', interpolation='nearest')
        plt.ylabel('chromatogram')
        plt.xlabel('peptide')
        plt.title('Heatmap of similarity for label true')
        plt.colorbar(label='Color Scale')
        plt.subplot(1, 2, 2)
        plt.imshow(sim_f, cmap='hot', interpolation='nearest')
        plt.ylabel('chromatogram')
        plt.xlabel('peptide')
        plt.title('Heatmap of similarity for label false')
        plt.colorbar(label='Color Scale')
        plt.savefig("./fig/" + "sim_" + fig_name, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))  # 设置图形的大小
        plt.subplot(1, 2, 1)
        plt.imshow(dist_t, cmap='hot', interpolation='nearest')
        plt.ylabel('chromatogram')
        plt.xlabel('peptide')
        plt.title('Heatmap of distance for label true')
        plt.colorbar(label='Color Scale')
        plt.subplot(1, 2, 2)
        plt.imshow(dist_f, cmap='hot', interpolation='nearest')
        plt.ylabel('chromatogram')
        plt.xlabel('peptide')
        plt.title('Heatmap of distance for label false')
        plt.colorbar(label='Color Scale')
        plt.savefig("./fig/" + "dist_"  + fig_name, dpi=300)
        plt.close()

    def plotROC(self, fpr, tpr, auc, fig_name):
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机猜测线
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig("./fig/" + fig_name, dpi=300)
        plt.close()

    def plotPR(self, recall, precision, auprc, fig_name):
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color='b', label=f'AUPRC = {auprc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig("./fig/" + fig_name, dpi=300)
        plt.close()