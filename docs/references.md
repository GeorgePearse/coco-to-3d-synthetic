# Research References

State-of-the-art methods in 3D reconstruction, Neural Radiance Fields (NeRF), and 3D Gaussian Splatting organized chronologically.

---

## Neural Radiance Fields (NeRF)

### 2020

#### **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (ECCV 2020)
- **Authors**: Ben Mildenhall, et al. (UC Berkeley, Google Research, UCSD)
- **Paper**: [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- **Code**: [bmild/nerf](https://github.com/bmild/nerf)
- **Project**: [nerf-website](https://www.matthewtancik.com/nerf)
- **Description**: The foundational paper that introduced Neural Radiance Fields. Represents scenes as continuous 5D functions (3D location + 2D viewing direction) optimized with volume rendering. Achieves photorealistic novel view synthesis but requires long training times (hours to days).

---

### 2021

#### **pixelNeRF: Neural Radiance Fields from One or Few Images** (CVPR 2021)
- **Authors**: Alex Yu, et al. (UC Berkeley)
- **Paper**: [arXiv:2012.02190](https://arxiv.org/abs/2012.02190)
- **Code**: [sxyu/pixel-nerf](https://github.com/sxyu/pixel-nerf)
- **Description**: Extends NeRF to work with one or few input images by conditioning on image features. Uses a fully-convolutional architecture that can generalize to novel scenes without per-scene optimization.

#### **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields** (ICCV 2021)
- **Authors**: Jonathan T. Barron, et al. (Google Research)
- **Paper**: [arXiv:2103.13415](https://arxiv.org/abs/2103.13415)
- **Code**: [google/mipnerf](https://github.com/google/mipnerf)
- **Description**: Addresses aliasing artifacts in NeRF by reasoning about conical frustums instead of rays. Significantly improves rendering quality, especially when rendering at different scales.

#### **BARF: Bundle-Adjusting Neural Radiance Fields** (ICCV 2021)
- **Authors**: Chen-Hsuan Lin, et al. (CMU, Meta AI)
- **Paper**: [arXiv:2104.06405](https://arxiv.org/abs/2104.06405)
- **Code**: [chenhsuanlin/bundle-adjusting-NeRF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)
- **Description**: Jointly optimizes NeRF and camera poses without requiring accurate camera calibration. Enables NeRF training from imperfect camera poses.

---

### 2022

#### **Instant Neural Graphics Primitives (Instant-NGP)** (SIGGRAPH 2022)
- **Authors**: Thomas Müller, et al. (NVIDIA)
- **Paper**: [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)
- **Code**: [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp)
- **Description**: Revolutionized NeRF training speed using multiresolution hash encoding. Reduces training time from hours to seconds while maintaining high quality. Includes CUDA implementation for real-time performance.

#### **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields** (CVPR 2022)
- **Authors**: Jonathan T. Barron, et al. (Google Research)
- **Paper**: [arXiv:2111.12077](https://arxiv.org/abs/2111.12077)
- **Code**: [google-research/multinerf](https://github.com/google-research/multinerf)
- **Description**: Extends Mip-NeRF to unbounded 360-degree scenes. Uses non-linear scene parameterization and online distillation for high-quality unbounded scene reconstruction.

#### **TensoRF: Tensorial Radiance Fields** (ECCV 2022)
- **Authors**: Anpei Chen, et al. (UCLA, Meta Reality Labs)
- **Paper**: [arXiv:2203.09517](https://arxiv.org/abs/2203.09517)
- **Code**: [apchenstu/TensoRF](https://github.com/apchenstu/TensoRF)
- **Description**: Models radiance fields as 4D tensors factorized into compact components. Achieves faster reconstruction with lower memory footprint compared to vanilla NeRF.

#### **NeRF-Studio: A Framework for Neural Radiance Field Development** (2022)
- **Authors**: Matthew Tancik, et al. (UC Berkeley, Luma AI)
- **Code**: [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- **Website**: [nerf.studio](https://docs.nerf.studio/)
- **Description**: Modular framework for NeRF research and development. Provides viewer, training infrastructure, and implementations of multiple NeRF variants. Industry-standard tool for NeRF experimentation.

---

### 2023

#### **Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields** (ICCV 2023)
- **Authors**: Jonathan T. Barron, et al. (Google Research)
- **Paper**: [arXiv:2304.06706](https://arxiv.org/abs/2304.06706)
- **Description**: Combines hash grids (Instant-NGP) with anti-aliasing (Mip-NeRF) for fast, high-quality reconstruction. Achieves state-of-the-art quality with efficient training times.

#### **K-Planes: Explicit Radiance Fields in Space, Time, and Appearance** (CVPR 2023)
- **Authors**: Sara Fridovich-Keil, et al. (UC Berkeley, Meta AI)
- **Paper**: [arXiv:2301.10241](https://arxiv.org/abs/2301.10241)
- **Code**: [sarafridov/K-Planes](https://github.com/sarafridov/K-Planes)
- **Description**: Factorizes 4D spacetime into six planes for efficient dynamic scene reconstruction. Enables modeling of time-varying scenes with explicit grid-based representation.

#### **NeRF-VAE: A Geometry Aware 3D Scene Generative Model** (ICML 2023)
- **Authors**: Adam R. Kosiorek, et al. (DeepMind)
- **Paper**: [arXiv:2104.00587](https://arxiv.org/abs/2104.00587)
- **Description**: Combines NeRF with variational autoencoders for generative 3D scene modeling. Learns disentangled representations of scene geometry and appearance.

---

### 2024

#### **LRM: Large Reconstruction Model for Single Image to 3D** (ICLR 2024)
- **Authors**: Yicong Hong, et al. (NVIDIA, Meta AI, University of Hong Kong)
- **Paper**: [arXiv:2311.04400](https://arxiv.org/abs/2311.04400)
- **Code**: [3DTopia/OpenLRM](https://github.com/3DTopia/OpenLRM) (open-source implementation)
- **Description**: 500M parameter transformer that predicts triplane NeRF from single images in ~5 seconds. Trained on 1M+ objects from Objaverse. State-of-the-art for fast single-image 3D reconstruction.

#### **SSDNeRF: Semantic-aware Single-stage Diffusion NeRF** (Apple, 2024)
- **Authors**: Apple Research
- **Paper**: [Research paper](https://machinelearning.apple.com/research/ssdnerf)
- **Description**: Unified single-stage training approach combining diffusion models with NeRF. End-to-end optimization for high-quality 3D generation with semantic awareness.

#### **MVDream: Multi-view Diffusion for 3D Generation** (2024)
- **Authors**: Yichun Shi, et al. (ByteDance)
- **Paper**: [arXiv:2308.16512](https://arxiv.org/abs/2308.16512)
- **Code**: [bytedance/MVDream](https://github.com/bytedance/MVDream)
- **Description**: Multi-view consistent diffusion model for text-to-3D. Trained on both 2D and 3D data to ensure view consistency during generation.

---

## 3D Gaussian Splatting

### 2023

#### **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (SIGGRAPH 2023)
- **Authors**: Bernhard Kerbl, et al. (Inria, Max Planck Institut)
- **Paper**: [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
- **Code**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **Project**: [gaussian-splatting-project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **Description**: Groundbreaking method representing scenes as 3D Gaussians instead of neural networks. Achieves real-time rendering (100+ FPS) with quality comparable to NeRF. Revolutionized real-time 3D reconstruction.

---

### 2024

#### **Splatter Image: Ultra-Fast Single-View 3D Reconstruction** (CVPR 2024)
- **Authors**: Stanislaw Szymanowicz, et al. (University of Oxford, Niantic)
- **Paper**: [arXiv:2312.13150](https://arxiv.org/abs/2312.13150)
- **Code**: [szymanowiczs/splatter-image](https://github.com/szymanowiczs/splatter-image)
- **Description**: Maps single image pixels directly to 3D Gaussians using feed-forward network. Ultra-fast inference for real-time single-view 3D reconstruction.

#### **InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models** (2024)
- **Authors**: Jiale Xu, et al. (Tencent ARC Lab)
- **Paper**: [arXiv:2404.07191](https://arxiv.org/abs/2404.07191)
- **Code**: [TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh)
- **Description**: Combines multiview diffusion with sparse-view reconstruction for high-quality textured meshes in ~10 seconds. Superior geometry and texture quality compared to earlier feed-forward methods.

#### **HumanSplat: Generalizable Single-Image Human Gaussian Splatting with Structure Priors** (NeurIPS 2024)
- **Authors**: Panwang Pan, et al.
- **Paper**: [arXiv:2406.12459](https://arxiv.org/abs/2406.12459)
- **Project**: [humansplat.github.io](https://humansplat.github.io/)
- **Description**: Human-specific Gaussian Splatting using structure priors for anatomically correct reconstruction. State-of-the-art for single-image human digitization with photorealistic novel views.

#### **FDGaussian: Fast Gaussian Splatting from Single Image via Geometric-aware Diffusion Model** (2024)
- **Authors**: Qijun Feng, et al.
- **Paper**: [arXiv:2403.10242](https://arxiv.org/abs/2403.10242)
- **Description**: Combines geometry-aware diffusion with Gaussian Splatting. Uses orthogonal plane decomposition and epipolar attention for view-consistent multi-view synthesis.

#### **LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation** (2024)
- **Authors**: Jiaxiang Tang, et al.
- **Paper**: [arXiv:2402.05054](https://arxiv.org/abs/2402.05054)
- **Code**: [3DTopia/LGM](https://github.com/3DTopia/LGM)
- **Description**: Multi-view Gaussian reconstruction model that replaces memory-intensive volume rendering. Fast rendering but with some multi-view inconsistency challenges.

#### **GSD: View-Guided Gaussian Splatting Diffusion for 3D Reconstruction** (ECCV 2024)
- **Authors**: Yuxuan Mu, et al.
- **Paper**: [arXiv:2407.04237](https://arxiv.org/abs/2407.04237)
- **Description**: Integrates diffusion models with Gaussian Splatting for high-quality object reconstruction. View-guided approach ensures consistency across generated views.

#### **TripoSR: Fast 3D Object Reconstruction from a Single Image** (2024)
- **Authors**: Stability AI, Tripo AI
- **Paper**: [arXiv:2403.02151](https://arxiv.org/abs/2403.02151)
- **Code**: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **Description**: Feed-forward 3D reconstruction model generating meshes in <0.5 seconds on A100. MIT licensed for commercial use. Excellent speed/quality balance for production use.

---

## Hybrid & Diffusion-Based Methods

### 2022

#### **DreamFusion: Text-to-3D using 2D Diffusion** (2022)
- **Authors**: Ben Poole, et al. (Google Research, UC Berkeley)
- **Paper**: [arXiv:2209.14988](https://arxiv.org/abs/2209.14988)
- **Code**: [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) (unofficial)
- **Description**: Pioneering text-to-3D method using Score Distillation Sampling (SDS) with 2D diffusion models as priors. Slow (~1.5 hours) but opened new research direction for text-to-3D generation.

---

### 2023

#### **Magic3D: High-Resolution Text-to-3D Content Creation** (CVPR 2023)
- **Authors**: Chen-Hsuan Lin, et al. (NVIDIA)
- **Paper**: [arXiv:2211.10440](https://arxiv.org/abs/2211.10440)
- **Description**: Two-stage coarse-to-fine text-to-3D optimization. Achieves 2x faster generation and 8x higher resolution than DreamFusion through sparse 3D hash grid and differentiable rendering.

#### **Zero-1-to-3: Zero-shot One Image to 3D Object** (ICCV 2023)
- **Authors**: Ruoshi Liu, et al. (Columbia University)
- **Paper**: [arXiv:2303.11328](https://arxiv.org/abs/2303.11328)
- **Code**: [cvlab-columbia/zero123](https://github.com/cvlab-columbia/zero123)
- **Description**: Diffusion model for novel view synthesis from single image with camera control. Enables controllable view generation for 3D reconstruction pipelines.

#### **Stable Zero123** (Stability AI, 2023)
- **Model**: [stabilityai/stable-zero123](https://huggingface.co/stabilityai/stable-zero123)
- **Description**: Improved version of Zero-1-to-3 with better training data and elevation conditioning. Open-source on Hugging Face, integrated with threestudio framework.

---

### 2024

#### **Wonder3D: Single Image to 3D using Cross-Domain Diffusion** (CVPR 2024)
- **Authors**: Xiaoxiao Long, et al.
- **Paper**: [arXiv:2310.15008](https://arxiv.org/abs/2310.15008)
- **Code**: [xxlong0/Wonder3D](https://github.com/xxlong0/Wonder3D)
- **Updates**: Wonder3D++ and Era3D (512x512 with auto focal length) released Dec 2024
- **Description**: Cross-domain diffusion for consistent geometry and texture generation. Produces high-detail textured meshes in 2-3 minutes.

#### **Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image** (NeurIPS 2024)
- **Authors**: Kailu Wu, et al. (AiuniAI)
- **Paper**: [arXiv:2405.20343](https://arxiv.org/abs/2405.20343)
- **Code**: [AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D)
- **Description**: Single-image to mesh in ~30 seconds using diffusion. High-fidelity diverse meshes that work on wild images with good generalization.

#### **CRM: Convolutional Reconstruction Model** (ECCV 2024)
- **Authors**: Zhaoxi Chen, et al. (Tsinghua University)
- **Paper**: [arXiv:2403.05034](https://arxiv.org/abs/2403.05034)
- **Code**: [thu-ml/CRM](https://github.com/thu-ml/CRM)
- **Description**: Feed-forward architecture generating textured meshes in ~10 seconds. Direct mesh output without intermediate representations. Good balance of speed and quality.

#### **SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion** (CVPR 2024)
- **Authors**: Hsuan-I Ho, et al.
- **Paper**: [arXiv:2311.15855](https://arxiv.org/abs/2311.15855)
- **Code**: [SiTH-Diffusion/SiTH](https://github.com/SiTH-Diffusion/SiTH)
- **Description**: Human-specific reconstruction using image-conditioned diffusion. Generates fully textured humans in ~2 minutes with high quality.

#### **SyncDreamer: Generating Multiview-Consistent Images from a Single-View Image** (2024)
- **Authors**: Yuan Liu, et al.
- **Paper**: [arXiv:2309.03453](https://arxiv.org/abs/2309.03453)
- **Description**: 3D-aware feature attention for multi-view consistent generation. Improves upon zero123 with better consistency across views.

#### **One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds** (NeurIPS 2023)
- **Authors**: Minghua Liu, et al. (HKUST, NVIDIA)
- **Paper**: [arXiv:2306.16928](https://arxiv.org/abs/2306.16928)
- **Description**: Fast single-image to 3D in 45 seconds using consistent multi-view generation without per-shape optimization. Quality trade-off for speed.

---

## Monocular Depth Estimation

### 2024 State-of-the-Art

#### **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data** (CVPR 2024)
- **Authors**: Lihe Yang, et al. (University of Hong Kong, TikTok)
- **Paper**: [arXiv:2401.10891](https://arxiv.org/abs/2401.10891)
- **Code**: [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- **Model**: [huggingface.co/depth-anything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything)
- **Description**: Winner of MDEC CVPR 2024 challenge with 48.8% improvement. Current state-of-the-art for relative depth estimation, trained on massive unlabeled data.

#### **Depth Pro: Sharp Monocular Metric Depth in Less Than a Second** (Apple, 2024)
- **Authors**: Apple Research
- **Paper**: [arXiv:2410.02073](https://arxiv.org/abs/2410.02073)
- **Description**: Ultra-fast (<1 second) metric depth estimation with sharp boundaries. Orders of magnitude faster than variable-resolution methods.

#### **Metric3D v2: A Versatile Monocular Geometric Foundation Model** (2024)
- **Authors**: Wei Yin, et al.
- **Paper**: [arXiv:2404.15506](https://arxiv.org/abs/2404.15506)
- **Description**: Rank #1 on multiple benchmarks. Trained on 16M+ images with thousands of camera models. Zero-shot metric depth and surface normals.

#### **MiDaS v3.1: Towards Robust Monocular Depth Estimation** (2022-2024)
- **Authors**: René Ranftl, et al. (Intel ISL)
- **Paper**: [arXiv:1907.01341](https://arxiv.org/abs/1907.01341) (original), ongoing updates
- **Code**: [isl-org/MiDaS](https://github.com/isl-org/MiDaS)
- **Description**: Robust relative depth estimation with model zoo. Industry-standard baseline used by many newer methods. Well-established and widely supported.

#### **DPT: Vision Transformers for Dense Prediction** (ICCV 2021)
- **Authors**: René Ranftl, et al. (Intel ISL)
- **Paper**: [arXiv:2103.13413](https://arxiv.org/abs/2103.13413)
- **Description**: Transformer-based dense prediction achieving superior accuracy metrics. Better geometric consistency than many alternatives, part of MiDaS model zoo.

---

## Multi-View Stereo and Matching

### 2024

#### **DUSt3R: Geometric 3D Vision Made Easy** (CVPR 2024)
- **Authors**: Shuzhe Wang, et al. (Naver Labs Europe)
- **Paper**: [arXiv:2312.14132](https://arxiv.org/abs/2312.14132)
- **Code**: [naver/dust3r](https://github.com/naver/dust3r)
- **Description**: Dense unconstrained stereo reconstruction without camera parameters. Foundational model approach that handles unconstrained images for 3D reconstruction.

#### **MASt3R: Matching and Stereo 3D Reconstruction** (ECCV 2024)
- **Authors**: Vincent Leroy, et al. (Naver Labs Europe)
- **Paper**: [arXiv:2406.09756](https://arxiv.org/abs/2406.09756)
- **Code**: [naver/mast3r](https://github.com/naver/mast3r)
- **Description**: Built on DUSt3R with dense local features. 30% improvement on challenging datasets. State-of-the-art for matching and metric 3D reconstruction.

---

## Human-Specific Reconstruction

### 2019-2020

#### **PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization** (ICCV 2019)
- **Authors**: Shunsuke Saito, et al. (Meta Reality Labs)
- **Paper**: [arXiv:1905.05172](https://arxiv.org/abs/1905.05172)
- **Code**: [shunsukesaito/PIFu](https://github.com/shunsukesaito/PIFu)
- **Description**: Pixel-aligned implicit function for high-resolution human digitization. Foundational work for human reconstruction from single images.

#### **PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization** (CVPR 2020)
- **Authors**: Shunsuke Saito, et al. (Meta Reality Labs)
- **Paper**: [arXiv:2004.00452](https://arxiv.org/abs/2004.00452)
- **Code**: [facebookresearch/pifuhd](https://github.com/facebookresearch/pifuhd)
- **Description**: Multi-level architecture (coarse + fine) for high-resolution clothed human reconstruction. No segmentation mask required. Industry standard for human avatars.

---

## Recommended Resources

### Frameworks and Tools

- **NeRF Studio**: [docs.nerf.studio](https://docs.nerf.studio/) - Complete NeRF development framework
- **threestudio**: [threestudio-project/threestudio](https://github.com/threestudio-project/threestudio) - Unified framework for 3D generation
- **Nerfacc**: [nerfacc](https://www.nerfacc.com/) - Efficient NeRF acceleration library

### Datasets

- **Objaverse**: [allenai/objaverse](https://objaverse.allenai.org/) - 1M+ 3D objects
- **MVImgNet**: Multi-view images for training
- **ScanNet++**: High-quality indoor scene scans
- **CO3Dv2**: Common Objects in 3D version 2

### Benchmarks and Challenges

- **MDEC Challenge (CVPR 2024)**: Monocular Depth Estimation Challenge
- **NeRF Synthetic**: Standard NeRF benchmark scenes
- **Mip-NeRF 360 Dataset**: Unbounded scene evaluation

---

## Latest Trends (2024-2025)

1. **Feed-Forward Models**: Shift from per-scene optimization to single-pass inference
2. **Gaussian Splatting Dominance**: Replacing NeRF for real-time applications
3. **Transformer Architectures**: Large reconstruction models (LRM) with 500M+ parameters
4. **Diffusion Integration**: Combining 2D diffusion priors with 3D representations
5. **Multi-View Consistency**: Better view-consistent generation (MVDream, SyncDreamer)
6. **Foundation Models**: Training on millions of objects for generalization
7. **Hybrid Representations**: Combining strengths of NeRF, Gaussians, and meshes
8. **Camera-Free Reconstruction**: DUSt3R/MASt3R eliminate need for camera parameters

---

*Last updated: November 2024*
