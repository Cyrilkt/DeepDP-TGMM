#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from sklearn.manifold import TSNE
import umap
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt


import numpy as np
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib as mpl
import torch.nn.functional as F
import pandas as pd


class PlotUtils:
    def __init__(self, hparams, logger=None, samples=None):
        self.mus_ind_merge = None
        self.mus_ind_split = None
        self.hparams = hparams
        self.logger = logger
        self.cmap = mpl.colors.ListedColormap(np.random.rand(100, 3))
        self.colors = None
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

    def plot_sphere_assignments(
        self,
        mus: torch.Tensor,                # [K,3]
        n_sub_list: list,                 # length K
        cluster_net: torch.nn.Module,
        subclustering_net: torch.nn.Module,
        conf_levels: list = [93,95,99],
        depth: int       = 200,
        tilt_deg: float  = 0,
        azim_deg: float  = 0,
        save_state_path: str = "cluster_state.pt"
    ):
        """
        For each main cluster k, plot the iso-frontier on S^2 of
        P(cluster=k|x) (solid) and P(sub|x) (dashed), and
        save minimal state for future plotting.
        """
        # 0) Save minimal cluster state
        torch.save({
            'mus': mus.cpu(),
            'n_sub_list': n_sub_list,
            'cluster_net_state': cluster_net.state_dict(),
            'sub_net_state': (
                subclustering_net.state_dict()
                if subclustering_net is not None else None
            )
        }, save_state_path)

        device = next(cluster_net.parameters()).device

        # helper: log-map to tangent-plane
        def log_map(p0, Q):
            d = np.clip(Q @ p0, -1, 1)
            gamma = np.arccos(d)
            sin_g = np.sin(gamma)
            sin_g[sin_g < 1e-8] = 1.0
            return ((gamma/sin_g)[:,None] * (Q - d[:,None]*p0[None,:]))

        # UV->3D mapping for contours
        def uv2xyz(segs_uv, R):
            out = []
            for seg in segs_uv:
                u_, v_ = seg[:,0], seg[:,1]
                xyz = np.stack([
                    np.cos(u_)*np.sin(v_),
                    np.sin(u_)*np.sin(v_),
                    np.cos(v_)
                ], axis=1)
                out.append(xyz @ R.T)
            return out

        # build sphere grid
        u = np.linspace(0, 2*np.pi, depth)
        v = np.linspace(0,     np.pi, depth)
        U, V = np.meshgrid(u, v)
        X = np.cos(U)*np.sin(V)
        Y = np.sin(U)*np.sin(V)
        Z = np.cos(V)
        Q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        K = mus.shape[0]
        for k in range(K):
            # 1) center
            p0 = mus[k].cpu().numpy()
            p0 /= np.linalg.norm(p0)

            # 2) tangent basis
            tmp = np.array([1,0,0])
            if np.allclose(p0, tmp): tmp = np.array([0,1,0])
            e1 = tmp - (tmp @ p0)*p0;  e1 /= np.linalg.norm(e1)
            e2 = np.cross(p0, e1)

            # 3) log-map Q->UV
            Vlog = log_map(p0, Q)
            UV   = np.stack([Vlog.dot(e1), Vlog.dot(e2)], axis=1)

            # 4) network probabilities
            with torch.no_grad():
                Qt = torch.from_numpy(Q).float().to(device)
                Pm = F.softmax(cluster_net(Qt), dim=1).cpu().numpy()[:,k]
                if subclustering_net is not None:
                    Ps_full = F.softmax(subclustering_net(Qt), dim=1).cpu().numpy()
                    offs = np.cumsum([0] + n_sub_list[:-1])
                    a,b = offs[k], offs[k] + n_sub_list[k]
                    Ps = Ps_full[:, a:b]
                else:
                    Ps = None

            # 5) confidence thresholds
            th_m = {p: np.percentile(Pm, p) for p in conf_levels}
            th_s = None if Ps is None else [
                {p: np.percentile(Ps[:,i], p) for p in conf_levels}
                for i in range(Ps.shape[1])
            ]

            # 6) extract 2D contours in UV plane
            def get_segs(Zvals, thr):
                fig, ax = plt.subplots()
                cs = ax.contour(U, V, Zvals.reshape(U.shape), levels=[thr])
                plt.close(fig)
                return cs.allsegs[0]

            segs_m_uv = {p: get_segs(Pm, th_m[p]) for p in conf_levels}
            segs_s_uv = None if Ps is None else [
                {p: get_segs(Ps[:,i], th_s[i][p]) for p in conf_levels}
                for i in range(Ps.shape[1])
            ]

            # 7) compute rotation p0->north + tilt + azim
            north = np.array([0,0,1])
            v = np.cross(p0, north); s = np.linalg.norm(v); c = p0 @ north
            Vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R_al = np.eye(3) if s<1e-8 else (np.eye(3) + Vx + Vx@Vx*((1-c)/(s*s)))
            a = np.deg2rad(tilt_deg)
            R_t = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
            b = np.deg2rad(azim_deg)
            R_z = np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1]])
            R = R_z @ R_t @ R_al

            Qr = Q @ R.T
            Xr = Qr[:,0].reshape(U.shape)
            Yr = Qr[:,1].reshape(U.shape)
            Zr = Qr[:,2].reshape(U.shape)

            # convert UV->XYZ segments
            segs_m = {p: uv2xyz(segs_m_uv[p], R) for p in conf_levels}
            segs_s = None if segs_s_uv is None else [
                {p: uv2xyz(segs_s_uv[i][p], R) for p in conf_levels}
                for i in range(len(segs_s_uv))
            ]

            # 8) plotting
            fig = plt.figure(figsize=(6,6))
            ax  = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.plot_surface(Xr, Yr, Zr, color='lightgray', alpha=0.15, edgecolor='none')

            # main cluster (solid)
            for p in conf_levels:
                for seg in segs_m[p]:
                    ax.plot(seg[:,0], seg[:,1], seg[:,2], color='black', lw=2,
                            label=f"Cl{k} {p}%" if seg is segs_m[p][0] else "")

            # subclusters (dashed)
            if segs_s is not None:
                cols = plt.cm.tab10(np.linspace(0,1,len(segs_s)))
                for i, sd in enumerate(segs_s):
                    for p in conf_levels:
                        for seg in sd[p]:
                            ax.plot(seg[:,0], seg[:,1], seg[:,2],
                                    color=cols[i], ls='--', lw=1.5,
                                    label=(f"Sub{i} {p}%") if (i,p)==(0,conf_levels[0]) else "")

            ax.view_init(elev=90, azim=0)
            ax.set_box_aspect([1,1,1])
            ax.axis('off')
            ax.legend(loc='upper right', fontsize='small')
            plt.tight_layout()
            # plt.show()  # avoid GUI in training
            plt.close(fig)
    #K.Cyril
    @staticmethod
    def debugging_visualize_embeddings(codes_dim, vae_means, vae_labels=None, current_epoch=None, UMAP=True, centers=None, fname=None,metric='cosine'):
        method='UMAP' if UMAP else 'TSNE'
        if codes_dim > 2:
            print(f"Transforming using {method}...")
            
            transformed_data = None
            #metric='cosine'
            if UMAP:
                umap_obj = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42,
                    metric=metric
                ).fit(vae_means.detach().cpu())
                transformed_data = umap_obj.embedding_
                
                if centers is not None and not isinstance(centers, np.ndarray):
                    centers = umap_obj.transform(centers.cpu())
                elif centers is not None and isinstance(centers, np.ndarray)  :
                    centers = umap_obj.transform(centers)
            else:
                tsne_obj = TSNE(n_components=2)
                if centers is not None :
                    # Combining the data and centers for t-SNE transformation
                    combined_data = np.vstack([vae_means.detach().cpu().numpy(), centers.cpu().numpy()])
                    all_transformed = tsne_obj.fit_transform(combined_data)
                    
                    # Separating the transformed data and centers
                    transformed_data = all_transformed[:-len(centers)]
                    centers = all_transformed[-len(centers):]
                else:
                    transformed_data = tsne_obj.fit_transform(vae_means.detach().cpu())
        else:
            print('BJR')
            transformed_data = vae_means.detach().cpu()
    
        fig = plt.figure(figsize=(16, 10))
        labels = vae_labels 
        #if ~labels.all():
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap="tab10")
        #else:
        #    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
            
        if centers is not None : #and :len(np.unique(labels)) == len(centers):
           from matplotlib.cm import get_cmap
           print("DRAWING CENTERS")
           n_centers = len(centers)
           cmap = get_cmap('tab10')  # Using the same colormap as your data points
           center_colors = [cmap(i) for i in range(n_centers)]
           plt.scatter(centers[:, 0], centers[:, 1], c=center_colors, marker='*', edgecolor='k', s=1100)
        
        plt.title(f"{method} embeddings, epoch {current_epoch}")
        plt.savefig(fname)
        plt.close(fig)
    
    @staticmethod
    def debugging_visualize_embeddingsv2(codes_dim, vae_means, vae_labels=None, current_epoch=None, UMAP=True, centers=None, fname=None, n_samples=None, M=1):
        method = 'UMAP' if UMAP else 'TSNE'
        
        if n_samples is None or n_samples >= len(vae_means):
            M = 1
    
        for i in range(M):
            current_data = vae_means
            current_labels = vae_labels
            current_centers = centers
    
            if n_samples and n_samples < len(vae_means):
                sample_indices = np.random.choice(len(vae_means), n_samples, replace=False)
                current_data = vae_means[sample_indices]
                if vae_labels is not None:
                    current_labels = vae_labels[sample_indices]
    
            if codes_dim >= 2:
                print(f"Transforming using {method} for map {i + 1}...")
    
                transformed_data = None
                if UMAP:
                    umap_obj = umap.UMAP(
                        n_neighbors=15,
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                    ).fit(current_data.detach().cpu())
                    transformed_data = umap_obj.embedding_
    
                    if centers is not None:
                        current_centers = umap_obj.transform(centers.cpu())
    
                else:
                    tsne_obj = TSNE(n_components=2)
                    if centers is not None:
                        # Combining the data and centers for t-SNE transformation
                        combined_data = np.vstack([current_data.detach().cpu().numpy(), centers.cpu().numpy()])
                        all_transformed = tsne_obj.fit_transform(combined_data)
                        
                        # Separating the transformed data and centers
                        transformed_data = all_transformed[:-len(centers)]
                        current_centers = all_transformed[-len(centers):]
                    else:
                        transformed_data = tsne_obj.fit_transform(current_data.detach().cpu())
            else:
                transformed_data = current_data.detach().cpu()
    
            fig = plt.figure(figsize=(16, 10))
            labels = current_labels 
            if ~labels.all():
                plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap="tab10")
            else:
                plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
                
            if current_centers is not None:
                from matplotlib.cm import get_cmap
                print("DRAWING CENTERS")
                n_centers = len(current_centers)
                cmap = get_cmap('tab10')  # Using the same colormap as your data points
                center_colors = [cmap(i) for i in range(n_centers)]
                plt.scatter(current_centers[:, 0], current_centers[:, 1], c=center_colors, marker='*', edgecolor='k', s=1100)
    
            plt.title(f"{method} embeddings, epoch {current_epoch}, map {i + 1}")
            plt.savefig(f"{fname}_map_{i + 1}")
            plt.close(fig)
    
    def visualize_embeddings(self, hparams, logger, codes_dim, vae_means, vae_labels=None, val_resp=None, current_epoch=None, UMAP=True, EM_labels=None, y_hat=None, stage="cluster_net_train", centers=None, training_stage='val'):
        method = "UMAP" if UMAP else "TSNE"
        if codes_dim > 2:
            if training_stage != "val_thesis" or (training_stage == "val_thesis" and not hasattr(self, 'val_embb')):
                print("Transforming using UMAP/TSNE...")
                if UMAP:
                    umap_obj = umap.UMAP(
                        n_neighbors=30,
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                        metric='cosine',
                    ).fit(vae_means.detach().cpu())
                    E = umap_obj.embedding_
                    if centers is not None:
                        centers = umap_obj.transform(centers.cpu())
                else:
                    E = TSNE(n_components=2).fit_transform(vae_means.detach().cpu())
        else:
            E = vae_means.detach().cpu()
        if val_resp is not None:
            if training_stage != "val_thesis":
                fig = plt.figure(figsize=(16, 10))
                plt.scatter(E[:, 0], E[:, 1], c=val_resp.argmax(-1), cmap="tab10")
                if centers is not None:
                    plt.scatter(centers[:, 0], centers[:, 1], c=np.arange(len(centers)), marker='*', edgecolor='k')
                plt.title(f"{method} embeddings, epoch {current_epoch}")
                from pytorch_lightning.loggers.base import DummyLogger
                if not isinstance(self.logger, DummyLogger):
                    logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net labels", fig)
                plt.close(fig)

            else:
                if hasattr(self, 'val_embb'):
                    E = self.val_embb
                else:
                    self.val_embb = E

                fig = plt.figure(figsize=(16, 10))
                plt.scatter(E[:, 0], E[:, 1], c=val_resp.argmax(-1), cmap="tab10")
                ax = plt.gca()
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                from pytorch_lightning.loggers.base import DummyLogger
                if not isinstance(self.logger, DummyLogger):
                    logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net labels new", fig)
                plt.close(fig)

        if y_hat is not None:
            fig = plt.figure(figsize=(16, 10))
            plt.scatter(E[:, 0], E[:, 1], c=y_hat, cmap="tab10")
            plt.title(f"{method} embeddings, epoch {current_epoch}")
            if fname is not None:
              plt.savefig(fname, bbox_inches="tight")

            from pytorch_lightning.loggers.base import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using net pseudo-labels", fig)
            plt.close(fig)

        fig = plt.figure(figsize=(16, 10))
        labels = vae_labels if EM_labels is None else EM_labels
        plt.scatter(E[:, 0], E[:, 1], c=labels, cmap="tab10")
        if centers is not None and len(np.unique(labels)) == len(centers):
            plt.scatter(centers[:, 0], centers[:, 1], c=sorted(np.unique(labels)), marker='*', edgecolor='k')
        if training_stage == "val_thesis":
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            plt.title(f"{method} embeddings, epoch {current_epoch}")

        from pytorch_lightning.loggers.base import DummyLogger
        if not isinstance(logger, DummyLogger):
            if EM_labels is None:
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using true labels", fig)
            else:
                logger.log_image(f"{stage}/{training_stage}/{method} embeddings using EM labels", fig)
        plt.close(fig)

    def visualize_embeddings_old(data, labels, use_pca_first=False):
        x = data.detach().cpu()
        if use_pca_first:
            print("Performing PCA...")
            pca_50 = PCA(n_components=50)
            data = pca_50.fit_transform(x)

        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=1, n_iter=1000, metric="cosine")
        data_2d = tsne.fit_transform(x)
        fig = plt.figure(figsize=(6, 5))
        num_classes = len(np.unique(labels))
        palette = np.array(sns.color_palette("hls", num_classes))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], lw=0, s=40, c=palette[labels])
        plt.legend()
        # plt.show()
        plt.close(fig)
        return fig

    def embed_to_2d(self, data):
        if data.shape[1] > 50:
            print("Performing PCA...")
            pca_50 = PCA(n_components=50)
            data = pca_50.fit_transform(data)
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=1, n_iter=1000, metric="cosine")
        data_2d = tsne.fit_transform(data)
        return data_2d

    def sklearn_make_ellipses(self, center, cov, ax, color, **kwargs):
        covariance = cov  # np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariance)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(np.abs(v))
        ell = mpl.patches.Ellipse(center, v[0], v[1], 180 + angle, color=color, **kwargs)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        return ell

    def draw_ellipse(self, x, y, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse((x, y), nsig * width, nsig * height, angle, **kwargs))

    def plot_clusters_colored_by_label(self, samples, y_gt, n_epoch, K):
        return self.plot_clusters_by_color(samples, y_gt, n_epoch, K, labels_type="gt")

    def plot_clusters_colored_by_net(self, samples, y_net, n_epoch, K):
        return self.plot_clusters_by_color(samples, y_net, n_epoch, K, labels_type="net")

    def plot_clusters_by_color(self, samples, labels, n_epoch, K, labels_type):
        fig = plt.figure(figsize=(16, 10))
        df = pd.DataFrame(columns=['x_pca', 'y_pca', 'label'])
        samples_pca = self.pca.transform(samples)
        df['x_pca'] = samples_pca[:, 0]
        df['y_pca'] = samples_pca[:, 1]
        df['label'] = labels
        df['label'] = df['label'].astype(str)
        sns.scatterplot(
            x="x_pca", y="y_pca",
            hue="label",
            # color=self.colors['']
            palette=sns.color_palette("hls", K),
            data=df,
            legend="full",
            alpha=0.3,
        )
        plt.title(f"Epoch {n_epoch}: pca-ed data colored by {labels_type} labels")
        return fig

    def plot_decision_regions(
        self,
        X,
        cluster_net,
        ax,
        y_gt
    ):
        X_min, X_max = X.min(axis=0).values, X.max(axis=0).values 
        arrays_for_meshgrid = [np.arange(X_min[d] - 0.1, X_max[d] + 0.1, 0.1) for d in range(X.shape[1])]
        xx, yy = np.meshgrid(*arrays_for_meshgrid)

        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        # horizontal stack vectors to create x1,x2 input for the model
        grid = np.hstack((r1,r2))
        yhat = cluster_net(torch.from_numpy(grid).float().to(self.device))
        yhat_maxed = yhat.max(axis=1).values.cpu()

        cont = ax.contourf(xx, yy, yhat_maxed.reshape(xx.shape), alpha=0.5, cmap="jet")

        ax.scatter(
            X[:, 0],
            X[:, 1],
            cmap="tab20",
            c=y_gt,
        )
        ax.set_title("Decision boundary \n Clusters are colored by GT labels")
        return cont

    def plot_clusters(
        self,
        ax_clusters,
        samples,
        labels,
        centers,
        covs,
        sub_center,
        sub_covs,
        mu_gt = None,
        n_epoch=None,
        alone=False,
    ):
        # expects to get samples, a 2-dim vector and labels are the true labels
        # centers are the centers that were found by the classifier

        if self.colors is None:
            # first time
            self.colors = torch.rand(self.hparams.init_k, 3)

        if alone:
            mu_gt = np.array([x.numpy() for x in mu_gt])
            if samples.shape[1] > 2:
                # perform PCA
                print("Performing PCA...")
                samples = self.pca.transform(samples)
                centers = self.pca.transform(centers)
                mu_gt = self.pca.transform(mu_gt)
                covs_pca, sub_covs_pca = [], []
                for cov in covs:
                    cov_diag = torch.tensor(cov).diag()
                    covs_pca.append(torch.eye(2) * self.pca.transform(cov_diag.reshape(1, -1)))
                covs = covs_pca
                if sub_covs is not None:
                    for sub_cov in sub_covs:
                        cov_diag = torch.tensor(sub_cov).diag()
                        sub_covs_pca.append(torch.eye(2) * self.pca.transform(cov_diag.reshape(1, -1)))
                    sub_covs = sub_covs_pca
            fig_clusters, ax_clusters = plt.subplots()

        # plot points colored by the given labels
        ax_clusters.scatter(
            samples[:, 0], samples[:, 1], c=self.colors[labels, :], s=40, alpha=0.5, zorder=1
        )

        # plot the gt centers and the net's centers
        if mu_gt:
            ax_clusters.plot(
                mu_gt[:, 0], mu_gt[:, 1], "g*", label="Real centers", markersize=15.0, zorder=2
            )
        ax_clusters.plot(
            centers[:, 0],
            centers[:, 1],
            "ko",
            label="net centers",
            markersize=12.0,
            alpha=0.6,
            zorder=3
        )

        # plot covs
        for i, center in enumerate(centers):
            ell = self.sklearn_make_ellipses(center=center, cov=covs[i], ax=ax_clusters, color=self.colors[i].numpy())
            ax_clusters.add_artist(ell)

        # plot net's subclusters
        if sub_center is not None:
            ax_clusters.scatter(
                sub_center[:, 0],
                sub_center[:, 1],
                marker='*',
                c=self.colors[np.arange(len(centers)).repeat(2)],
                edgecolors='k',
                label="net subcenters",
                s=100.0,
                alpha=1,
                zorder=4
            )
            for j, sub in enumerate(sub_center):
                ax_clusters.text(sub[0]+0.03, sub[1], str(j % 2), c='k', fontsize=12)  # self.colors[j // 2].numpy(), fontsize=12)

            for i in range(len(sub_center)):
                ell = self.sklearn_make_ellipses(center=sub_center[i], cov=sub_covs[i], ax=ax_clusters, color=self.colors[i//2].numpy(), ls="--", fill=False)
                ax_clusters.add_artist(ell)

        ax_clusters.set_title("Net centers and covariances \n Clusters are colored by net's assignments")
        # ax_clusters.legend()
        if alone:
            ax_clusters.set_title(
                f"Epoch {n_epoch}: Clusters colored by net's assignments"
            )
            return fig_clusters

    def plot_cluster_and_decision_boundaries(
        self,
        samples,
        labels,
        gt_labels,
        net_centers,
        net_covs,
        n_epoch,
        cluster_net,
        gt_centers=None,
        net_sub_centers=None,
        net_sub_covs=None,
    ):
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 8))
        # fig.tight_layout()
        if gt_centers:
            gt_centers = np.array([x.numpy() for x in gt_centers])

        # set aspect ratio
        _min, _max = samples.min(axis=0).values, samples.max(axis=0).values 
        (ax_clusters, ax_boundaries) = axes

        self.plot_clusters(
            ax_clusters=ax_clusters,
            samples=samples,
            labels=labels,
            centers=net_centers,
            covs=net_covs,
            mu_gt=gt_centers,
            sub_center=net_sub_centers,
            sub_covs=net_sub_covs,
        )

        cont_for_color_bar = self.plot_decision_regions(
                X=samples, cluster_net=cluster_net, ax=ax_boundaries, y_gt=gt_labels)

        for ax in axes:
            ax.set_xlim([_min[0], _max[0]])
            ax.set_ylim([_min[1], _max[1]])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            
        cbar = fig.colorbar(cont_for_color_bar, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("Max network response", rotation=270, labelpad=10, y=0.45)

        # fig.suptitle(f"Epoch: {n_epoch}", fontsize=14, weight="bold")

        import os
        if not os.path.exists("./imgs/"):
            os.makedirs("./imgs/")
            os.makedirs("./imgs/clusters/")
            os.makedirs("./imgs/decision_boundary/")
        fig.savefig(f"./imgs/{n_epoch}.png")

        # save just the clusters fig
        extent_clus = ax_clusters.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        extent_bound = ax_boundaries.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        if cluster_net.training:
            fig.savefig(
                f"./imgs/clusters/{n_epoch}.png",
                bbox_inches=extent_clus,
            )
            fig.savefig(
                f"./imgs/decision_boundary/{n_epoch}.png",
                bbox_inches=extent_bound,
            )

        plt.close()
        return

    def plot_weights_histograms(self, K, pi, start_sub_clustering, current_epoch, pi_sub, for_thesis=False):
        fig = plt.figure(figsize=(10, 3))
        ind = np.arange(K)
        plt.bar(ind, pi, label="clusters' weights", align="center", alpha=0.3)
        if start_sub_clustering <= current_epoch and pi_sub is not None:
            pi_sub_1 = pi_sub[0::2]
            pi_sub_2 = pi_sub[1::2]
            plt.bar(ind, pi_sub_1, align="center", label="sub cluster 1")
            plt.bar(
                ind, pi_sub_2, align="center", bottom=pi_sub_1, label="sub cluster 2"
            )

        plt.xlabel("Clusters inds")
        plt.ylabel("Normalized weights")
        plt.title(f"Epoch {current_epoch}: Clusters weights")
        if for_thesis:
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            plt.legend()
        return fig

    def plot_cov_eig_values(self, covs_to_plot, epoch):
        for i in range(len(covs_to_plot)):
            e, _ = torch.torch.linalg.eig(covs_to_plot[i])
            e = torch.real(e)
            fig = plt.figure(figsize=(16, 10))
            plt.plot(range(len(e)), sorted(e, reverse=True))
            plt.title(f"the eigenvalues of cov {i} to be split, epoch {epoch}")
            plt.xlabel("Eigenvalues inds")
            plt.ylabel("Eigenvalues")
            self.logger.log_image(f"cluster_net_train/train/epoch {epoch}/eigenvalues_cov_{i}", fig)
            plt.close(fig)

    def update_colors(self, split, split_inds, merge_inds):
        if split:
            self.update_colors_split(split_inds)
        else:
            self.update_colors_merge(merge_inds)


    def update_colors_split(self, mus_ind_split):
        mask = torch.zeros(len(self.colors), dtype=bool)
        mask[mus_ind_split.flatten()] = 1
        colors_not_split = self.colors[torch.logical_not(mask)]
        colors_split = self.colors[mask].repeat(1, 2).view(-1, 3)
        colors_split[1::2] = torch.rand(len(colors_split[1::2]), 3)
        self.colors = torch.cat([colors_not_split, colors_split])

    def update_colors_merge(self, mus_ind_merge):
        mask = torch.zeros(len(self.colors))
        mask[mus_ind_merge.flatten()] = 1
        colors_not_merged = self.colors[torch.logical_not(mask)]
        # take all the non merges clusters' colors and the color of the first index of a merge pair
        self.colors = torch.cat(
            [colors_not_merged, self.colors[mus_ind_merge[:, 0]]]
        )


