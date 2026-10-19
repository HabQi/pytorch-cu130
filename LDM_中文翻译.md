# 利用潜在扩散模型进行高分辨率图像合成

**原题：** High-Resolution Image Synthesis with Latent Diffusion Models  
**作者：** Robin Rombach、Andreas Blattmann、Dominik Lorenz、Patrick Esser、Björn Ommer。前两位作者贡献相同。  
**机构：** 德国慕尼黑大学与海德堡大学 IWR；Runway ML。  
**版本：** arXiv:2112.10752v2 [cs.CV]，2022 年 4 月 13 日。  
**原文件：** [LDM.pdf](LDM.pdf)，共 45 页：正文第 1–9 页，参考文献第 10–13 页，附录第 14–45 页。  
**代码：** <https://github.com/CompVis/latent-diffusion>

> 本译稿包含正文、附录 A–H、全部 34 幅图、18 张表和 109 条参考文献。章节、图表、公式与文献引用沿用原编号。图片取自本地 PDF，图内英文标识保留，图注译为中文；表格转写为 Markdown，并提供原表图片供核对。参考文献保留原文书目信息。文中“最先进”等表述均指论文发表时的比较结果。
>
> 术语：latent diffusion model（LDM）译为“潜在扩散模型”，latent space 译为“潜在空间”，perceptual compression 译为“感知压缩”，semantic compression 译为“语义压缩”，cross-attention 译为“交叉注意力”，classifier-free guidance 译为“无分类器引导”，inpainting 译为“图像修复”，super-resolution 译为“超分辨率”。模型名、数据集名和代码标识保留原文。$256^2$ 等表示 $256\times256$ 像素；$f$ 是空间下采样倍数；表中 M、B 分别表示百万和十亿，k/K 表示千；“—”对应原表的缺省或未报告项。

> **阅读注释（全文主线）：** 下文新增的“阅读注释”是辅助理解的解释或推导，与论文译文及原文勘误用的“译注”区分。阅读时可抓住三个问题：为什么先压缩图像、如何在压缩表示上学习生成、怎样把文本等条件送入生成过程。对应的核心组件依次是自编码器、去噪 UNet 和条件编码器／交叉注意力。

## 摘要

扩散模型（DM）将图像形成过程分解为去噪自编码器的顺序应用，在图像数据及其他领域实现了最先进的合成结果。
此外，其形式允许通过引导机制控制图像生成过程，无须重新训练。
然而，这些模型通常直接在像素空间中运行，因此，强大扩散模型的优化往往需要数百个 GPU 日，而顺序执行的模型求值也使推理成本高昂。
为了在有限计算资源下训练扩散模型，同时保留其质量与灵活性，我们将其应用于强大预训练自编码器的潜在空间。
与此前工作不同，在这种表示上训练扩散模型，首次能够在降低复杂度和保留细节之间达到近乎最优的平衡，大幅提高视觉保真度。
通过在模型架构中引入交叉注意力层，我们将扩散模型转变为强大且灵活的生成器，可以接受文本或边界框等通用条件输入，并以卷积方式实现高分辨率合成。
我们的潜在扩散模型（LDM）在图像修复和类别条件图像合成上取得新的最先进分数，并在文本到图像合成、无条件图像生成和超分辨率等多项任务上表现出很强的竞争力，同时显著降低了相对于像素空间扩散模型的计算需求。

## 1 引言

图像合成是计算机视觉近年来进展最为显著的领域之一，同时也是计算需求最大的领域之一。
尤其是复杂自然场景的高分辨率合成，目前主要依赖对基于似然的模型进行扩展；其中的自回归（AR）Transformer 可能包含数十亿参数[66,67]。
相比之下，GAN[3,27,40] 虽然结果出色，但其成果主要局限于变化相对有限的数据，因为对抗学习过程难以直接扩展到复杂、多模态分布的建模。
近期，由层次化去噪自编码器构成的扩散模型[82]，在图像合成[30,85]及其他领域[7,45,48,57]取得了令人瞩目的结果，并在类别条件图像合成[15,31]和超分辨率[72]上达到了最先进水平。
此外，与其他类型的生成模型[19,46,69]不同，即使是无条件扩散模型，也能方便地应用于图像修复、着色[85]或基于笔画的合成[53]等任务。
作为基于似然的模型，它们不会出现 GAN 的模式坍塌和训练不稳定现象；同时，通过充分利用参数共享，它们能够对自然图像的高度复杂分布进行建模，而无须像自回归模型那样使用数十亿参数[67]。

![图 1：不同空间压缩程度下的重建对比](LDM_译文图像/figure_01.png)

**图 1：通过较温和的下采样，提高可实现质量的上限。** 由于扩散模型对空间数据具有出色的归纳偏置，我们无须像其他潜在空间生成模型那样进行强烈的空间下采样；使用适当的自编码模型，仍可大幅降低数据维度，见第 3 节。图像来自 DIV2K[1] 验证集，以 $512^2$ 像素进行评估。$f$ 表示空间下采样倍数。重建 FID（R-FID）[29]和 PSNR 在 ImageNet 验证集[12]上计算，另见表 8。图中依次为输入、本文方法（$f=4$，PSNR 27.4、R-FID 0.58）、DALL-E（$f=8$，PSNR 22.8、R-FID 32.01）、VQGAN（$f=16$，PSNR 19.9、R-FID 4.98）。

> **阅读注释（先分清重建与生成）：** 图 1 比较的是把真实图像编码后再解码得到的重建结果，检验自编码器能保留多少信息；后文的生成 FID 则评价从随机噪声出发生成的图像分布。重建好为后续生成提供了良好的表示基础，但不能单凭 R-FID 判断生成质量。“质量上限”在这里强调表示瓶颈，并不意味着 R-FID 与生成 FID 之间存在一个可直接套用的严格不等式。

**让高分辨率图像合成更易获得。** 扩散模型属于基于似然的模型，其覆盖模式的行为使其容易将过多容量，以及相应的计算资源，花在数据中不可感知的细节建模上[16,73]。
尽管重新加权的变分目标[30]试图通过降低初始去噪步骤的采样频率来处理这个问题，扩散模型的计算需求仍然很高，因为训练和评估需要在高维 RGB 图像空间中反复进行函数求值及梯度计算。
例如，训练最强大的扩散模型往往需要数百个 GPU 日（文献[15]中为 150–1000 个 V100 日）；对输入空间的含噪版本进行重复求值，也使推理成本高昂，在单张 A100 GPU 上生成 5 万个样本约需 5 天[15]。
这给研究社区和普通用户带来两方面影响：首先，训练这样的模型需要只有少数研究者才能获得的大量计算资源，并留下很大的碳足迹[65,86]；其次，已经训练好的模型在评估时，同样需要消耗大量时间和内存，因为相同的模型架构必须顺序运行许多步（例如文献[15]中的 25–1000 步）。

要提高这类强大模型的可及性，同时减少其显著的资源消耗，就需要一种同时降低训练和采样计算复杂度的方法。
因此，在不损害性能的前提下降低扩散模型的计算需求，是提高其可及性的关键。

**转向潜在空间。** 我们的方法始于对已经训练好的像素空间扩散模型的分析：图 2 展示了一个训练后模型的率失真权衡。
与任何基于似然的模型一样，其学习过程大致可以分为两个阶段：第一阶段是感知压缩，移除高频细节，但尚未学到多少语义变化；第二阶段中，真正的生成模型学习数据的语义与概念组合，即语义压缩。
因此，我们希望先找到一个感知等价、但计算上更合适的空间，再在其中训练用于高分辨率图像合成的扩散模型。

![图 2：感知压缩与语义压缩](LDM_译文图像/figure_02.png)

**图 2：感知压缩与语义压缩示意。** 数字图像的大多数比特对应不可感知的细节。扩散模型虽然可以通过最小化相应损失项来抑制这些没有语义意义的信息，但训练中的梯度，以及训练和推理中的神经网络骨干，仍需要在所有像素上进行计算，造成多余计算和不必要的高昂优化、推理成本。我们提出以潜在扩散模型作为有效的生成模型，并使用一个单独、温和的压缩阶段，仅去除不可感知的细节。数据与图像来自文献[30]。横轴为码率（比特/维），纵轴为失真（RMSE）。

> **阅读注释（两种压缩与两个训练阶段）：** “感知压缩”侧重去掉视觉上不重要的细节，同时保留可重建的内容；“语义压缩”侧重学习对象、布局和外观等数据规律。LDM 用第一阶段自编码器承担前一部分，再在其表示上训练扩散模型。这里的两个训练阶段，与一次生成中的多个去噪时间步是不同层次的概念。

按照常见做法[11,23,66,67,96]，我们将训练分为两个独立阶段。
首先训练一个自编码器，提供低维、因而高效的表示空间，并使其与数据空间在感知上等价。
与此前工作[23,66]不同，关键在于我们无须依赖过度的空间压缩，因为扩散模型是在学习得到的潜在空间中训练的，其计算量随空间维度增长的性质更好。
降低复杂度后，也可以通过一次网络前向计算，高效地将潜在空间中的表示转换成图像。
我们将得到的这类模型称为**潜在扩散模型（LDM）**。

这种方法的一个显著优点是，通用的自编码阶段只需训练一次，因此可以在多次扩散模型训练中重复使用，或用于探索完全不同的任务[81]。

> **阅读注释（哪些参数在哪个阶段学习）：** 第一阶段训练图像编码器 $\mathcal E$ 和解码器 $\mathcal D$；第二阶段固定这两个模型，训练去噪网络。对于式（3）的条件模型，还联合训练条件编码器 $\tau_\theta$。因此，“联合训练条件编码器与 UNet”不表示把图像自编码器也一起重新训练。
这使得我们能够高效探索大量用于各种图像到图像、文本到图像任务的扩散模型。
针对后者，我们设计了一种架构，将 Transformer 连接到扩散模型的 UNet 骨干[71]，支持任意类型的基于词元的条件机制，见第 3.3 节。

我们的贡献如下：

1. 与纯 Transformer 方法[23,66]相比，本方法能够更平稳地扩展到更高维的数据，因此既可以使用比此前工作更忠实、更细致的重建所需的压缩程度（见图 1），也可以高效应用于百万像素图像的高分辨率合成。
2. 在多项任务（无条件图像合成、图像修复、随机超分辨率）和多个数据集上取得具有竞争力的性能，同时显著降低计算成本。与像素空间扩散方法相比，推理成本也显著降低。
3. 此前工作[93]同时学习编码器／解码器架构与基于得分的先验；我们的方法无须在重建能力与生成能力之间进行精细的权重平衡。这保证了极高保真度的重建，并且只需对潜在空间施加很弱的正则化。
4. 对于超分辨率、图像修复和语义合成等具有稠密条件的任务，本模型能够以卷积方式应用，生成约 $1024^2$ 像素的大幅、一致图像。
5. 我们设计了基于交叉注意力的通用条件机制，支持多模态训练，并用其训练类别条件、文本到图像和布局到图像模型。
6. 我们发布了预训练潜在扩散模型与自编码模型：<https://github.com/CompVis/latent-diffusion>；除训练扩散模型外，它们还可能复用于其他多种任务[81]。

## 2 相关工作

**用于图像合成的生成模型。** 图像的高维特性给生成建模带来了独特挑战。
生成对抗网络（GAN）[27]能高效生成感知质量良好的高分辨率图像[3,42]，但难以优化[2,28,54]，也难以捕获完整的数据分布[55]。
相比之下，基于似然的方法侧重良好的密度估计，因此优化行为更稳定。
变分自编码器（VAE）[46]和基于流的模型[18,19]能够高效合成高分辨率图像[9,44,92]，但样本质量不及 GAN。
自回归模型（ARM）[6,10,94,95]虽然在密度估计上表现强大，但计算需求高的架构[97]和顺序采样过程使其局限于低分辨率图像。
由于图像的像素表示包含几乎不可感知的高频细节[16,73]，最大似然训练会将不成比例的容量用于这些细节的建模，导致训练时间很长。
为了扩展到更高分辨率，一些两阶段方法[23,67,101,103]使用自回归模型对压缩后的图像潜在空间建模，而非直接对原始像素建模。

近期，扩散概率模型（DM）[82]在密度估计[45]和样本质量[15]方面都取得了最先进的结果。
当底层神经网络骨干采用 UNet[15,30,71,85]时，这些模型的生成能力来自其与图像类数据归纳偏置的自然契合。
使用重新加权的目标[30]训练，通常可以得到最佳合成质量。
此时，扩散模型相当于有损压缩器，可以在图像质量与压缩能力之间进行权衡。
然而，在像素空间评估和优化这些模型的缺点是推理速度慢、训练成本极高。
前者可以通过先进采样策略[47,75,84]和层次化方法[31,93]得到部分缓解，但对高分辨率图像数据进行训练始终需要昂贵的梯度计算。
我们提出的 LDM 在维度更低的压缩潜在空间中工作，同时解决了这两个缺点，使训练的计算成本更低，推理更快，且几乎不降低合成质量（见图 1）。

**两阶段图像合成。** 为缓解单一生成方法的不足，已有大量研究[11,23,67,70,101,103]通过两阶段方法，结合不同方法的优势，构建更高效、性能更好的模型。
VQ-VAE[67,101]使用自回归模型，在离散化潜在空间上学习具有表现力的先验。
文献[66]通过学习离散图像表示与文本表示的联合分布，将这种方法扩展到文本到图像生成。
更一般地，文献[70]使用条件可逆网络，为不同领域的潜在空间之间提供通用转换。
与 VQ-VAE 不同，VQGAN[23,103]在第一阶段使用对抗目标和感知目标，使自回归 Transformer 能扩展到更大图像。
然而，为使自回归模型训练可行而需要的高压缩率，以及由此采用的数十亿可训练参数[23,66]，限制了这类方法的总体性能；降低压缩程度又会带来高昂计算成本[23,66]。
我们的工作避免了这种权衡，因为 LDM 的卷积骨干使其能够更平缓地扩展到更高维的潜在空间。
因此，我们能够自由选择压缩程度，在学习强大的第一阶段模型、不把过多感知压缩留给生成扩散模型，以及保证高保真重建之间取得最佳平衡（见图 1）。

已有方法将编码／解码模型与基于得分的先验联合学习[93]或分别学习[80]；前者仍需要在重建与生成能力之间进行困难的权重平衡[11]，且性能不如我们的方法（第 4 节）；后者则主要关注人脸等结构高度规整的图像。

## 3 方法

为了降低用于高分辨率图像合成的扩散模型的训练计算需求，我们注意到：虽然扩散模型可以通过减少对应损失项的采样来忽略感知上无关的细节[30]，它们仍需要在像素空间中执行昂贵的函数求值，因而消耗大量计算时间和能源。
我们提出将压缩学习阶段与生成学习阶段显式分离，以绕过这一缺点（见图 2）。
为此，我们使用自编码模型学习一个与图像空间感知等价、但计算复杂度显著降低的空间。

这种方法具有若干优点：第一，离开高维图像空间后，采样在低维空间进行，扩散模型的计算效率大幅提升。
第二，利用扩散模型从 UNet 架构[71]继承的归纳偏置，使其对具有空间结构的数据特别有效，从而减轻了此前方法[23,66]所需的激进、损害质量的压缩要求。
第三，我们得到通用压缩模型，其潜在空间既可用于训练多个生成模型，也可用于单图像 CLIP 引导合成[25]等其他下游应用。

### 3.1 感知图像压缩

我们的感知压缩模型基于此前工作[23]，由一个自编码器组成，通过感知损失[106]与基于图像块[33]的对抗目标[20,23,103]联合训练。
这通过强制局部真实性，使重建结果位于图像流形上，并避免仅使用像素空间损失（例如 $L_2$ 或 $L_1$ 目标）时产生的模糊。

> **阅读注释（为什么需要感知损失与对抗损失）：** 逐像素损失关心对应位置的数值是否相同；当细节存在多种合理解释时，降低均方误差可能得到平滑的折中结果。感知损失比较特征层面的相似性，图像块判别器约束局部纹理的真实性。它们帮助压缩模型保留视觉效果，但不保证所有细节都与输入逐像素一致。

具体而言，给定 RGB 空间中的图像 $x\in\mathbb R^{H\times W\times3}$，编码器 $\mathcal E$ 将其编码为潜在表示 $z=\mathcal E(x)$，解码器 $\mathcal D$ 从潜变量重建图像，得到 $\tilde x=\mathcal D(z)=\mathcal D(\mathcal E(x))$，其中 $z\in\mathbb R^{h\times w\times c}$。
关键是编码器将图像下采样 $f=H/h=W/w$ 倍；我们研究不同下采样倍数 $f=2^m$，其中 $m\in\mathbb N$。

> **阅读注释（潜变量是什么，节省了多少维度）：** $z$ 是由图像编码器学到的二维特征网格，每个位置含 $c$ 个通道；这些通道通常不再具有 RGB 的直接含义。空间位置数从 $HW$ 减为 $HW/f^2$，总标量数之比为 $c/(3f^2)$。例如 $256\times256\times3$ 经 $f=4,c=3$ 编码后为 $64\times64\times3$，标量数变为原来的 $1/16$。实际速度还取决于网络通道数、注意力和解码开销，不能直接等同于 16 倍加速。

为了避免潜在空间出现任意大的方差，我们实验了两种正则化方式。
第一种称为 **KL-reg.**，类似 VAE[46,69]，对学习得到的潜变量施加一个很弱的、朝向标准正态分布的 KL 惩罚；第二种称为 **VQ-reg.**，在解码器内使用向量量化层[96]。
后者可以理解为一个将量化层吸收到解码器中的 VQGAN[23]。
由于后续扩散模型利用的是学习得到的潜在空间 $z=\mathcal E(x)$ 的二维结构，我们可以采用相对温和的压缩率，并获得很好的重建。
相比之下，此前工作[23,66]依赖对学习空间 $z$ 施加任意一维顺序，再以自回归方式对其分布建模，忽略了 $z$ 的大量内在结构。
因此，我们的压缩模型能够更好地保留 $x$ 的细节（见表 8）。
完整目标与训练细节见附录。

> **阅读注释（KL 正则化与 VQ 正则化）：** KL 方案让每张图像对应一个可采样的连续后验，并用很弱的惩罚约束它；VQ 方案用码本约束表示。本文 VQ 方案的扩散模型仍处理量化之前的连续特征，量化放在解码器端。弱 KL 正则化也不意味着真实图像的全部潜变量已经严格服从标准正态；它们的分布仍需要后续扩散模型学习，见 G 节。

### 3.2 潜在扩散模型

**扩散模型。** 扩散模型[82]通过逐步去噪一个服从正态分布的变量，学习数据分布 $p(x)$，这相当于学习一条长度为 $T$ 的固定马尔可夫链的反向过程。
在图像合成中，最成功的模型[15,30,72]依赖针对 $p(x)$ 的变分下界的重新加权变体，其形式与去噪得分匹配[85]相对应。
这些模型可以理解为一系列权重相等的去噪自编码器 $\epsilon_\theta(x_t,t)$，$t=1,\ldots,T$；训练它们预测输入 $x_t$ 的去噪版本，其中 $x_t$ 是输入 $x$ 的含噪版本。
对应的目标可以简化为（见附录 B）：

$$
L_{\mathrm{DM}}=\mathbb E_{x,\epsilon\sim\mathcal N(0,1),t}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right],
\tag{1}
$$

其中 $t$ 从 $\{1,\ldots,T\}$ 中均匀采样。

> **阅读注释（式 1 实际在预测什么）：** 按这个损失的参数化，网络输出的是所加入噪声 $\epsilon$ 的估计，目标是让 $\epsilon_\theta(x_t,t)$ 接近训练时已知的 $\epsilon$。预测噪声后可反推出干净图像估计，见式（14）。不同 $t$ 共用同一组参数 $\theta$，并将时间步作为输入；训练通常随机抽取时间步即可计算损失，无须为每个样本完整执行一遍反向生成链。

**潜在表示的生成建模。** 通过训练好的感知压缩模型 $\mathcal E$ 和 $\mathcal D$，我们获得了高效、低维的潜在空间，其中高频、不可感知的细节已经被抽象去除。
与高维像素空间相比，这个空间更适合基于似然的生成模型：它们现在既能专注于数据中重要的语义比特，也能在维度更低、计算效率更高的空间中训练。
此前工作依赖高度压缩的离散潜在空间中的自回归注意力 Transformer 模型[23,66,103]；我们则能够利用模型针对图像提供的归纳偏置。
这包括主要使用二维卷积层构建底层 UNet，并使用重新加权的界进一步将目标聚焦于感知上最相关的比特。此时目标为：

$$
L_{\mathrm{LDM}}:=\mathbb E_{\mathcal E(x),\epsilon\sim\mathcal N(0,1),t}
\left[\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2\right].
\tag{2}
$$

> **阅读注释（式 2 的一次训练样本）：** 先取得 $z_0=\mathcal E(x)$，再采样 $t$ 和标准高斯噪声 $\epsilon$，按附录 B 的记号构造 $z_t=\alpha_t z_0+\sigma_t\epsilon$。把 $(z_t,t)$ 送入 UNet，计算预测噪声与真实噪声的平方误差。$z_0$、$z_t$、$\epsilon$ 与 UNet 输出的张量形状相同。这个构造可直接跳到任意 $t$，无需逐步加噪到该时刻。

模型的神经网络骨干 $\epsilon_\theta(\circ,t)$ 采用以时间为条件的 UNet[71]。
由于前向过程固定，训练时可由 $\mathcal E$ 高效获得 $z_t$；从 $p(z)$ 得到的样本，只需通过 $\mathcal D$ 一次，即可解码至图像空间。

> **阅读注释（生成流程与“一次解码”）：** 生成时从近似标准高斯噪声 $z_T$ 出发，在潜在空间中反复调用 UNet 和采样更新，得到 $z_0$，最后执行 $x=\mathcal D(z_0)$。“一次”指最后的解码器前向计算，整条扩散采样链仍包含多次网络调用。无条件生成或文本到图像生成不需要先编码一张目标图像。

![图 3：潜在扩散模型与条件机制](LDM_译文图像/figure_03.png)

**图 3：** 我们通过拼接或更通用的交叉注意力机制为 LDM 提供条件，见第 3.3 节。图中从左到右为像素空间、潜在空间和条件输入；条件可来自语义图、文本、其他表示或图像。图例包括去噪步骤、交叉注意力、切换、跳跃连接和拼接。

### 3.3 条件机制

与其他类型的生成模型[56,83]类似，扩散模型原则上能够对 $p(z\mid y)$ 形式的条件分布建模。
可以通过条件去噪自编码器 $\epsilon_\theta(z_t,t,y)$ 实现，从而通过文本[68]、语义图[33,61]或其他图像到图像转换任务[34]的输入 $y$ 来控制合成过程。
然而，在图像合成中，将扩散模型的生成能力与类别标签[15]、输入图像的模糊版本[72]之外的条件类型结合，迄今仍是一个探索不足的研究领域。

我们通过在底层 UNet 骨干中加入交叉注意力机制[97]，将扩散模型变成更灵活的条件图像生成器；这种机制已经被证明能有效学习基于注意力的多种输入模态模型[35,36]。
为预处理来自不同模态的 $y$（例如语言提示词），我们引入领域专用编码器 $\tau_\theta$，将 $y$ 投影为中间表示 $\tau_\theta(y)\in\mathbb R^{M\times d_\tau}$，再通过交叉注意力层映射到 UNet 的中间层：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)\cdot V,
$$

$$
Q=W_Q^{(i)}\cdot\varphi_i(z_t),\qquad
K=W_K^{(i)}\cdot\tau_\theta(y),\qquad
V=W_V^{(i)}\cdot\tau_\theta(y).
$$

这里，$\varphi_i(z_t)\in\mathbb R^{N\times d_\epsilon^i}$ 表示实现 $\epsilon_\theta$ 的 UNet 的一个展平后的中间表示；$W_V^{(i)}\in\mathbb R^{d\times d_\epsilon^i}$、$W_Q^{(i)}\in\mathbb R^{d\times d_\tau}$ 和 $W_K^{(i)}\in\mathbb R^{d\times d_\tau}$ 为可学习投影矩阵[36,97]。
可视化示意见图 3。

> **阅读注释（交叉注意力中 Q、K、V 的作用）：** 图像特征产生查询 $Q$，文本等条件特征产生键 $K$ 和值 $V$。按“每行一个词元”的写法，$Q\in\mathbb R^{N\times d}$、$K\in\mathbb R^{M\times d}$，所以 $QK^\top$ 的形状是 $N\times M$：每个图像位置对各条件词元分配注意力权重，再汇总对应的值。它允许不同空间位置使用不同条件信息，词元与图像区域之间也不必是一一对应的。

> **译注：** 此处矩阵维度按原文保留。原文所列 $W_Q$、$W_V$ 的输入维度与前面的 $Q$、$V$ 定义不一致；按其映射对象理解，应分别对应 UNet 特征维度和条件特征维度。实际实现还需根据行向量／列向量约定处理转置。

随后，基于图像与条件的配对数据，我们用以下目标学习条件 LDM：

$$
L_{\mathrm{LDM}}:=\mathbb E_{\mathcal E(x),y,\epsilon\sim\mathcal N(0,1),t}
\left[\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\|_2^2\right],
\tag{3}
$$

其中 $\tau_\theta$ 和 $\epsilon_\theta$ 均通过式（3）联合优化。
这种条件机制很灵活，因为 $\tau_\theta$ 可以由领域专用的模型来参数化；例如，当 $y$ 是文本提示词时，可使用不带掩码的 Transformer[97]（见第 4.3.1 节）。

> **阅读注释（式 3 中的两类表示）：** $z_t$ 来自图像编码器的潜在空间，是被加噪并逐步生成的对象；$\tau_\theta(y)$ 来自条件编码器，为去噪提供信息。两者即使都被称为“编码”或“表示”，来源与作用也不同。式（3）相对于式（2）的核心变化，是让同一个噪声预测任务额外依赖条件，而不是把训练目标改成直接回归文本或类别。

![图 4：不同数据集上的生成样本](LDM_译文图像/figure_04.png)

**图 4：** 分别在 CelebA-HQ[39]、FFHQ[41]、LSUN-Churches[102]、LSUN-Bedrooms[102] 和类别条件 ImageNet[12] 上训练的 LDM 生成样本；分辨率均为 $256\times256$。放大查看效果更佳，更多样本见附录。

## 4 实验

LDM 为不同图像模态提供了灵活、计算上可行的扩散式图像合成手段，下文将通过实验证明这一点。
不过，首先我们分析相对于像素空间扩散模型，本模型在训练和推理上的收益。
有趣的是，尽管 VQ 正则化第一阶段模型的重建能力略逊于其连续表示对应模型（见表 8），在 VQ 正则化潜在空间中训练的 LDM 有时反而获得更好的样本质量。
不同第一阶段正则化方案对 LDM 训练及其向大于 $256^2$ 分辨率泛化能力的影响，可视化比较见附录 D.1。
本节全部结果的架构、实现、训练与评估细节见 E.2。

### 4.1 感知压缩的权衡

本节分析不同下采样倍数 $f\in\{1,2,4,8,16,32\}$ 下 LDM 的行为，简称 LDM-$f$；其中 LDM-1 对应像素空间扩散模型。
为构建可比较的实验条件，本节全部实验均使用单张 NVIDIA A100，所有模型训练相同步数且参数量相同。
本节所比较 LDM 使用的第一阶段模型的超参数与重建性能见表 8。
图 6 展示了 ImageNet[12] 数据集上的类别条件模型，在 200 万训练步内，样本质量随训练进度的变化。
我们看到：一方面，LDM-{1,2} 的较小下采样倍数导致训练进展缓慢；另一方面，过大的 $f$ 会使保真度在相对较少训练步后停止改善。
回顾前述分析（图 1、2），我们认为前者是因为将大部分感知压缩留给了扩散模型，后者则是第一阶段过强的压缩导致信息损失，限制了可实现的质量。
LDM-{4–16} 在效率与感知保真度之间取得良好平衡：训练 200 万步后，像素空间扩散模型 LDM-1 与 LDM-8 的 FID[29] 相差高达 38。

在图 7 中，我们比较了在 CelebA-HQ[39] 和 ImageNet 上训练的模型，使用 DDIM 采样器[84]在不同去噪步数下的采样速度，并将其与 FID 分数[29]对照。
LDM-{4–8} 优于感知压缩与概念压缩比例不合适的模型。
尤其相对于像素空间的 LDM-1，它们在显著提高样本吞吐量的同时，获得了低得多的 FID。
对于 ImageNet 等复杂数据集，需要降低压缩程度以避免损害质量。
总体而言，LDM-4 和 LDM-8 为获得高质量合成结果提供了最佳条件。

> **阅读注释（如何理解最优压缩程度）：** $f$ 太小，扩散模型要处理的空间网格很大；$f$ 太大，自编码器先丢掉了过多信息，后续生成受表示能力限制。本文的 $f=4$ 或 $8$ 是特定架构、数据集和实验预算下的结果。图 6 按训练步数比较，图 17 按 V100 日比较；相同步数不代表相同耗时，应结合两类曲线理解效率收益。

![图 6：不同下采样倍数的训练进度](LDM_译文图像/figure_06.png)

**图 6：** 分析 ImageNet 数据集上不同下采样倍数 $f$ 的类别条件 LDM，在 200 万训练步内的训练情况。像素空间的 LDM-1 比较大下采样倍数模型 LDM-{4–16} 所需训练时间明显更长。LDM-32 这样的过度感知压缩会限制总体样本质量。所有模型在相同计算预算下使用单张 NVIDIA A100 训练。结果采用 100 步 DDIM[84]、$\kappa=0$ 得到。左图为 FID 随训练进度的变化，右图为 Inception 分数随训练进度的变化。

![图 7：采样吞吐量与质量](LDM_译文图像/figure_07.png)

**图 7：** 比较不同压缩程度的 LDM 在 CelebA-HQ（左）和 ImageNet（右）上的表现。沿每条线从右向左，不同标记分别表示 DDIM 的 $\{10,20,50,100,200\}$ 个采样步。虚线表示 200 步时的 FID，显示出 LDM-{4–8} 的强大性能。FID 使用 5000 个样本评估。所有模型均在一张 A100 上训练，CelebA 为 50 万步，ImageNet 为 200 万步。

### 4.2 使用潜在扩散进行图像生成

我们在 CelebA-HQ[39]、FFHQ[41]、LSUN-Churches 和 LSUN-Bedrooms[102] 上训练 $256^2$ 图像的无条件模型，并使用 FID[29]和 Precision/Recall[50]分别评估样本质量及其对数据流形的覆盖程度。
结果汇总于表 1。

> **阅读注释（FID、Precision、Recall 各看什么）：** FID 比较真实图像与生成图像在特征空间中的分布统计，通常越低越好；Precision 反映生成结果有多少落在真实数据的有效区域附近，Recall 反映覆盖了多少真实数据变化。这里的 Precision、Recall 是生成分布指标，不是类别预测的查准率、查全率。综合阅读它们，比仅凭一项指标更容易判断质量与多样性的取舍。
在 CelebA-HQ 上，我们取得了 5.11 的新最先进 FID，优于此前基于似然的模型和 GAN。
我们也优于将潜在扩散模型与第一阶段联合训练的 LSGM[93]。
相比之下，我们在固定空间中训练扩散模型，避免了在重建质量与潜在空间先验学习之间进行权衡的困难，见图 1、2。
除 LSUN-Bedrooms 外，我们在所有数据集上都优于此前的扩散方法；在该数据集上，我们使用 ADM[15] 一半的参数、四分之一的训练资源，得到的分数仍接近 ADM（见附录 E.3.5）。
此外，LDM 在 Precision 和 Recall 上持续优于基于 GAN 的方法，证实了覆盖模式的似然训练目标相对于对抗方法的优势。
图 4 展示了各数据集的定性结果。

**表 1：无条件图像合成的评估指标。** [原表](LDM_译文图像/table_01_original.png)

| 数据集（均为 256×256） | 方法 | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|
| CelebA-HQ | DC-VAE[63] | 15.8 | — | — |
| CelebA-HQ | VQGAN+T.[23]（k=400） | 10.2 | — | — |
| CelebA-HQ | PGGAN[39] | 8.0 | — | — |
| CelebA-HQ | LSGM[93] | 7.22 | — | — |
| CelebA-HQ | UDM[43] | 7.16 | — | — |
| CelebA-HQ | LDM-4（本文，500 步†） | 5.11 | 0.72 | 0.49 |
| FFHQ | ImageBART[21] | 9.57 | — | — |
| FFHQ | U-Net GAN（+数据增强）[77] | 10.9（7.6） | — | — |
| FFHQ | UDM[43] | 5.54 | — | — |
| FFHQ | StyleGAN[41] | 4.16 | 0.71 | 0.46 |
| FFHQ | ProjectedGAN[76] | 3.08 | 0.65 | 0.46 |
| FFHQ | LDM-4（本文，200 步） | 4.98 | 0.73 | 0.50 |
| LSUN-Churches | DDPM[30] | 7.89 | — | — |
| LSUN-Churches | ImageBART[21] | 7.32 | — | — |
| LSUN-Churches | PGGAN[39] | 6.42 | — | — |
| LSUN-Churches | StyleGAN[41] | 4.21 | — | — |
| LSUN-Churches | StyleGAN2[42] | 3.86 | — | — |
| LSUN-Churches | ProjectedGAN[76] | 1.59 | 0.61 | 0.44 |
| LSUN-Churches | LDM-8*（本文，200 步） | 4.02 | 0.64 | 0.52 |
| LSUN-Bedrooms | ImageBART[21] | 5.51 | — | — |
| LSUN-Bedrooms | DDPM[30] | 4.9 | — | — |
| LSUN-Bedrooms | UDM[43] | 4.57 | — | — |
| LSUN-Bedrooms | StyleGAN[41] | 2.35 | 0.59 | 0.48 |
| LSUN-Bedrooms | ADM[15] | 1.90 | 0.66 | 0.51 |
| LSUN-Bedrooms | ProjectedGAN[76] | 1.52 | 0.61 | 0.34 |
| LSUN-Bedrooms | LDM-4（本文，200 步） | 2.95 | 0.66 | 0.48 |

CelebA-HQ 的结果引自文献[43,63,100]，FFHQ 的结果引自文献[42,43]。†：原表的 N-s 表示使用 DDIM 采样器[84]采样 N 步。*：在 KL 正则化潜在空间中训练。更多结果见附录。

### 4.3 条件潜在扩散

#### 4.3.1 用于 LDM 的 Transformer 编码器

通过为 LDM 引入基于交叉注意力的条件机制，我们使其能够接受此前扩散模型尚未探索过的多种条件模态。
对于文本到图像建模，我们在 LAION-400M[78] 上训练一个由语言提示词提供条件、具有 14.5 亿参数的 KL 正则化 LDM。
我们使用 BERT 分词器[14]，并将 $\tau_\theta$ 实现为 Transformer[97]，推断潜在编码，再通过多头交叉注意力将其映射至 UNet（第 3.3 节）。
将语言表示学习与视觉合成这两类领域专用模型相结合，得到的强大模型能够很好地泛化到复杂的用户自定义提示词，参见图 8、5。
在定量分析中，我们遵循此前工作，在 MS-COCO[51] 验证集上评估文本到图像生成；本模型优于强大的自回归方法[17,66]和 GAN 方法[109]，见表 2。
我们注意到，使用无分类器扩散引导[32]能大幅提升样本质量，使引导后的 LDM-KL-8-G 在显著减少参数量的同时，达到近期文本到图像合成最先进自回归模型[26]和扩散模型[59]相近的水平。

> **阅读注释（条件输入与无分类器引导）：** 交叉注意力负责把条件送入网络；无分类器引导（CFG）则在采样时组合有条件和空条件预测。采用一种常见的强度约定，可写为 $\hat\epsilon=\epsilon_{\mathrm{uncond}}+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})$：$s=1$ 为普通条件预测，$s>1$ 放大条件影响。它要求模型能处理空条件，通常通过训练时随机去掉条件获得；不需要额外分类器。增强引导可能提高条件符合度，同时改变多样性。这是辅助说明式，其他文献的强度参数可能采用不同约定。
为进一步分析交叉注意力条件机制的灵活性，我们还在 OpenImages[49] 上训练基于语义布局的图像合成模型，再在 COCO[4] 上微调，见图 8。
定量评估与实现细节见 D.3。

最后，遵循此前工作[3,15,21,23]，我们在表 3、图 4 和 D.4 中评估第 4.1 节表现最好的类别条件 ImageNet 模型，其中 $f\in\{4,8\}$。
我们在显著降低计算需求与参数量的同时，优于最先进扩散模型 ADM[15]，见表 18。

![图 5：文本到图像生成样本](LDM_译文图像/figure_05.png)

**图 5：** 文本到图像合成模型 LDM-8（KL）针对用户自定义提示词生成的样本。该模型在 LAION[78] 数据库上训练。使用 200 步 DDIM、$\eta=1.0$，并使用强度 $s=10.0$ 的无条件引导[32]。图中标注模型参数量为 14.5 亿。

从左到右的提示词译文：

1. 一块写着“Latent Diffusion（潜在扩散）”的路牌。
2. 毕加索风格的僵尸。
3. 一种一半是老鼠、一半是章鱼的动物的图像。
4. 一个略有意识的神经网络的插画。
5. 一幅松鼠吃汉堡的画。
6. 一把看起来像章鱼的椅子的水彩画。
7. 一件印着“I love generative models!（我爱生成模型！）”的衬衫。

**表 2：在 256×256 的 MS-COCO[51] 上评估文本条件图像合成。** 使用 250 步 DDIM[84]，本模型以明显更少的参数量，达到与最新扩散方法[59]和自回归方法[26]相近的水平。†／* 的数值分别来自文献[109]／[26]。[原表](LDM_译文图像/table_02_original.png)

| 方法 | FID ↓ | IS ↑ | 参数量 | 采样／引导设置 |
|---|---:|---:|---:|---|
| CogView†[17] | 27.10 | 18.20 | 4B | 自排序；拒绝率 0.017 |
| LAFITE†[109] | 26.94 | 26.02 | 75M | — |
| GLIDE*[59] | 12.24 | — | 6B | 277 步 DDIM；无分类器引导[32]，s=3 |
| Make-A-Scene*[26] | 11.84 | — | 4B | 自回归模型的无分类器引导[98]，s=5 |
| LDM-KL-8 | 23.31 | 20.03±0.33 | 1.45B | 250 步 DDIM |
| LDM-KL-8-G* | 12.63 | 30.29±0.42 | 1.45B | 250 步 DDIM；无分类器引导[32]，s=1.5 |

![图 8：COCO 布局到图像生成](LDM_译文图像/figure_08.png)

**图 8：** 在 COCO[4] 上使用 LDM 进行布局到图像合成，见第 4.3.1 节。定量评估见附录 D.3。

**表 3：类别条件 ImageNet LDM 与近期最先进方法的比较。** 任务为 ImageNet[12] 上的类别条件图像生成。更多基线的详细比较见 D.4、表 10 和附录 F。c.f.g. 表示文献[32]提出的、强度为 $s$ 的无分类器引导。[原表](LDM_译文图像/table_03_original.png)

| 方法 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ | 参数量 | 设置 |
|---|---:|---:|---:|---:|---:|---|
| BigGAN-deep[3] | 6.95 | 203.6±2.6 | 0.87 | 0.28 | 340M | — |
| ADM[15] | 10.94 | 100.98 | 0.69 | 0.63 | 554M | 250 步 DDIM |
| ADM-G[15] | 4.59 | 186.7 | 0.82 | 0.52 | 608M | 250 步 DDIM |
| LDM-4（本文） | 10.56 | 103.49±1.24 | 0.71 | 0.62 | 400M | 250 步 DDIM |
| LDM-4-G（本文） | 3.60 | 247.67±5.59 | 0.87 | 0.48 | 400M | 250 步；无分类器引导[32]，s=1.5 |

#### 4.3.2 超越 256² 的卷积采样

通过将空间对齐的条件信息拼接到 $\epsilon_\theta$ 的输入，LDM 可以成为高效、通用的图像到图像转换模型。
我们用这种方式训练语义合成、超分辨率（第 4.4 节）和图像修复（第 4.5 节）模型。
对于语义合成，我们使用风景图像与语义图的配对数据[23,61]，将下采样后的语义图与 $f=4$ 模型的图像潜在表示拼接（VQ 正则化，见表 8）。
我们在 $256^2$ 输入分辨率上训练，即从 $384^2$ 图像中裁剪，但发现本模型能泛化到更大分辨率，以卷积方式评估时可以生成百万像素级图像（见图 9）。
利用这一特性，我们也将第 4.4 节超分辨率模型和第 4.5 节修复模型应用于生成 $512^2$ 至 $1024^2$ 的大图。
在此应用中，由潜在空间尺度决定的信噪比会显著影响结果。
在 D.1 中，我们分别在 $f=4$ 模型直接提供的潜在空间（KL 正则化，见表 8）及按各分量标准差重新缩放后的潜在空间中学习 LDM，展示这一现象。
后者结合无分类器引导[32]，还使文本条件模型 LDM-KL-8-G 能直接合成大于 $256^2$ 的图像，如图 13。

> **阅读注释（为何能够生成大于训练尺寸的图像）：** 对支持可变空间尺寸的网络，可以扩大潜在噪声网格，并提供相应尺寸、空间对齐的条件，再进行去噪。它保留了训练时学到的局部处理方式，但更大画面的全局布局未必自动连贯。本文为这种泛化提供了实验结果，并在附录 C、D.1 讨论引导和信噪比的影响；不能据此假设任意尺寸下质量都相同。

![图 9：在更大分辨率上卷积采样](LDM_译文图像/figure_09.png)

**图 9：** 在 $256^2$ 分辨率上训练的 LDM，可在风景图像语义合成等空间条件任务中泛化到更大分辨率，此处为 $512\times1024$。见第 4.3.2 节。

### 4.4 使用潜在扩散进行超分辨率

通过拼接低分辨率图像作为直接条件，可以高效训练用于超分辨率的 LDM（见第 3.3 节）。
在第一个实验中，我们遵循 SR3[72]，将图像退化固定为四倍下采样的双三次插值，并按照 SR3 的数据处理流程在 ImageNet 上训练。
我们使用在 OpenImages 上预训练的 $f=4$ 自编码模型（VQ 正则化，见表 8），将低分辨率条件 $y$ 与 UNet 输入拼接，也就是令 $\tau_\theta$ 为恒等映射。
定性与定量结果（图 10、表 5）显示出具有竞争力的性能：LDM-SR 的 FID 优于 SR3，而 SR3 的 IS 更好。
简单的图像回归模型取得最高 PSNR 与 SSIM；然而，这些指标与人类感知的一致性不强[106]，相比未完全对齐的高频细节，它们更偏好模糊结果[72]。

> **阅读注释（超分辨率为何会出现指标分歧）：** 同一低分辨率图像可能对应多种高分辨率纹理。像素回归倾向于产生折中结果，生成模型则可能采样出清晰、合理但与真实纹理不同的细节。因此，“看起来更真实”和“与这张真实图像逐像素更接近”可能给出不同排序；表 5 同时报告感知分布指标与重建指标，正是为了展示这种取舍。
此外，我们进行用户研究，比较像素空间基线与 LDM-SR。
按照 SR3[72]，我们向受试者展示一张位于两张高分辨率图像中间的低分辨率图像，并询问偏好。
表 4 的结果证实了 LDM-SR 的良好性能。
通过事后引导机制[15]可以提高 PSNR 和 SSIM；我们使用感知损失实现这种基于图像的引导器，见 D.6。

由于双三次退化过程对不遵循这一预处理的图像泛化不佳，我们还通过更多样的退化训练通用模型 LDM-BSR，结果见 D.6.1。

![图 10：ImageNet 超分辨率对比](LDM_译文图像/figure_10.png)

**图 10：** 在 ImageNet 验证集上进行 $64\to256$ 超分辨率。LDM-SR 更善于渲染真实纹理，SR3 则能合成更连贯的精细结构。更多样本和局部裁剪见附录。SR3 结果来自文献[72]。从左到右为双三次插值、LDM-SR、SR3。

**表 4：用户研究。** 任务 1：展示真实图像与生成图像，让受试者选择偏好的图像。任务 2：让受试者在两张生成图像之间选择。细节见 E.3.6。[原表](LDM_译文图像/table_04_original.png)

| 用户研究 | ImageNet 超分：Pixel-DM（f=1） | ImageNet 超分：LDM-4 | Places 修复：LaMa[88] | Places 修复：LDM-4 |
|---|---:|---:|---:|---:|
| 任务 1：相对于真实图像的偏好率 ↑ | 16.0% | 30.4% | 13.6% | 21.0% |
| 任务 2：偏好分数 ↑ | 29.4% | 70.6% | 31.9% | 68.1% |

**表 5：ImageNet 验证集上的四倍上采样结果（256²）。** †：FID 参考特征在验证集上计算；‡：参考特征在训练集上计算；吞吐量在一张 NVIDIA A100 上测量。[原表](LDM_译文图像/table_05_original.png)

| 方法 | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | 参数量 | 样本/秒 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 图像回归[72] | 15.2 | 121.1 | 27.9 | 0.801 | 625M | 不适用 |
| SR3[72] | 5.2 | 180.1 | 26.4 | 0.762 | 625M | 不适用 |
| LDM-4（本文，100 步） | 2.8†/4.8‡ | 166.3 | 24.4±3.8 | 0.69±0.14 | 169M | 4.62 |
| LDM-4（本文，大模型，100 步） | 2.4†/4.3‡ | 174.9 | 24.7±4.1 | 0.71±0.15 | 552M | 4.5 |
| LDM-4（本文，50 步，引导） | 4.4†/6.4‡ | 153.7 | 25.8±3.7 | 0.74±0.12 | 184M | 0.38 |

### 4.5 使用潜在扩散进行图像修复

图像修复是用新内容填充图像被掩码区域的任务，目的是修补损坏的图像部分，或替换已有但不需要的内容。
我们评估这种通用条件图像生成方法，与针对该任务专门设计的最先进方法相比如何。
评估遵循近期修复模型 LaMa[88] 的协议；LaMa 引入了基于快速傅里叶卷积[8]的专用架构。
在 Places[108] 上的具体训练与评估协议见 E.2.2。

> **阅读注释（修复是在学习条件分布）：** 修复的目标可理解为：给定可见内容和掩码，生成与上下文相容的缺失区域。被遮住的内容往往存在多种合理答案，改变采样噪声即可产生不同补全。因此，生成结果与被遮挡的真实图像有差异，并不自动意味着补全不合理；但“合理”也不等于恢复了原本发生的真实内容。

首先，我们分析第一阶段不同设计选择的影响。
具体而言，比较 LDM-1（即像素空间条件扩散模型）与 LDM-4 的修复效率；LDM-4 分别使用 KL 和 VQ 正则化，另包括第一阶段完全不使用注意力的 VQ-LDM-4（见表 8），后者可降低高分辨率解码的 GPU 显存需求。
为保证可比性，我们固定所有模型的参数量。
表 6 报告了 $256^2$ 和 $512^2$ 分辨率下的训练与采样吞吐量、每轮总训练时间，以及 6 轮后验证集上的 FID。
总体上，相比像素空间扩散模型，潜在扩散模型至少加速 2.7 倍，同时 FID 至少改善 1.6 倍。

**表 6：图像修复效率评估。** †：由于 GPU 设置、批大小不同，数值与图 7 有所差异，见附录。[原表](LDM_译文图像/table_06_original.png)

| 模型（正则化类型） | 训练吞吐量（样本/秒） | 256 采样吞吐量† | 512 采样吞吐量† | 训练+验证（小时/轮） | 第 6 轮 FID@2k |
|---|---:|---:|---:|---:|---:|
| LDM-1（无第一阶段） | 0.11 | 0.26 | 0.07 | 20.66 | 24.74 |
| LDM-4（KL，有注意力） | 0.32 | 0.97 | 0.34 | 7.66 | 15.21 |
| LDM-4（VQ，有注意力） | 0.33 | 0.97 | 0.34 | 7.04 | 14.99 |
| LDM-4（VQ，无注意力） | 0.35 | 0.99 | 0.36 | 6.66 | 15.95 |

表 7 与其他修复方法的比较表明，我们带注意力的模型在 FID 衡量的总体图像质量上优于文献[88]。
完整未掩码图像与生成样本之间的 LPIPS 略高于文献[88]。
我们认为这是因为文献[88]只生成一个结果，更倾向于恢复平均图像，而 LDM 产生的结果具有多样性，参见图 21。
此外，用户研究（表 4）中受试者也更偏好我们的结果。

基于这些初步结果，我们还在不带注意力、采用 VQ 正则化的第一阶段潜在空间中训练了更大的扩散模型（表 7 中的 big）。
遵循文献[15]，该扩散模型的 UNet 在三个特征层级使用注意力层，采用 BigGAN[3]残差块进行上下采样，参数量从 215M 增至 387M。
训练后，我们发现 $256^2$ 与 $512^2$ 分辨率生成样本的质量存在差异，推测是额外注意力模块导致的。
不过，在 $512^2$ 分辨率上微调半轮后，模型能够适应新的特征统计量，并在图像修复上取得新的最先进 FID（表 7 中 big、w/o attn、w/ ft，见图 11）。

> **阅读注释（表中“无注意力”指哪一部分）：** 本段大模型的“w/o attn”指第一阶段自编码器不含注意力；扩散 UNet 本身仍按本段描述包含注意力层。第一阶段与扩散骨干是两个独立组件，阅读配置时应先确认缩写修饰的是哪一个。这里高分辨率微调的收益，也说明架构能够接受更大输入与在该尺寸上已充分适应，是两个不同问题。

![图 11：对象移除效果](LDM_译文图像/figure_11.png)

**图 11：** 使用大模型并经过微调的修复模型（big、w/ ft）进行对象移除的定性结果。更多结果见图 22。左列为输入，右列为结果。

**表 7：Places[108] 测试图像中 3 万张 512×512 裁剪图上的修复性能。** “遮挡 40–50%”列报告较难样本的指标，这些样本有 40–50% 图像区域需要修复。†：由于无法获取文献[88]使用的原测试集，结果在本文测试集上重新计算。[原表](LDM_译文图像/table_07_original.png)

| 方法 | 遮挡 40–50%：FID ↓ | 遮挡 40–50%：LPIPS ↓ | 全部样本：FID ↓ | 全部样本：LPIPS ↓ |
|---|---:|---:|---:|---:|
| LDM-4（本文，大模型，有微调） | 9.39 | 0.246±0.042 | 1.50 | 0.137±0.080 |
| LDM-4（本文，大模型，无微调） | 12.89 | 0.257±0.047 | 2.40 | 0.142±0.085 |
| LDM-4（本文，有注意力） | 11.87 | 0.257±0.042 | 2.15 | 0.144±0.084 |
| LDM-4（本文，无注意力） | 12.60 | 0.259±0.041 | 2.37 | 0.145±0.084 |
| LaMa[88]† | 12.31 | 0.243±0.038 | 2.23 | 0.134±0.080 |
| LaMa[88] | 12.0 | 0.24 | 2.21 | 0.14 |
| CoModGAN[107] | 10.4 | 0.26 | 1.82 | 0.15 |
| RegionWise[52] | 21.3 | 0.27 | 4.75 | 0.15 |
| DeepFill v2[104] | 22.1 | 0.28 | 5.20 | 0.16 |
| EdgeConnect[58] | 30.5 | 0.28 | 8.37 | 0.16 |

## 5 局限与社会影响

**局限。** 虽然相对于像素空间方法，LDM 显著降低了计算需求，但其顺序采样过程仍慢于 GAN。
此外，在需要高精度时，使用 LDM 可能并不合适：尽管 $f=4$ 自编码模型的图像质量损失非常小（见图 1），其重建能力仍可能成为要求像素空间精细准确性任务的瓶颈。
我们认为，超分辨率模型（第 4.4 节）在这方面已经受到一定限制。

> **阅读注释（压缩带来的精度边界）：** 若两张图像被编码为相同的潜变量，解码器就无法仅凭该潜变量判断原图究竟是哪张。生成先验能补出符合数据规律的细节，却不能保证这些细节等于压缩前丢失的信息。这解释了为什么感知质量良好的压缩仍可能影响需要精确位置或纹理的任务。

**社会影响。** 图像等媒体的生成模型是一把双刃剑：一方面，它们支持多种创意应用；尤其像我们这样降低训练和推理成本的方法，有潜力使更多人获得这项技术，并让更多人参与探索。
另一方面，这也意味着制作和传播被篡改数据、散布虚假信息和垃圾信息变得更加容易。
尤其是蓄意操纵图像，即“深度伪造”，是这一背景下的常见问题，女性受到的影响尤为严重[13,24]。

生成模型还可能泄露训练数据[5,90]；当数据包含敏感信息或个人信息，且未经明确同意就被收集时，这一点尤其值得关注。
不过，这在多大程度上也适用于图像扩散模型，目前尚未被充分理解。

最后，深度学习模块往往会复制或加剧数据中已有的偏差[22,38,91]。
尽管扩散模型比 GAN 等方法更好地覆盖了数据分布，我们这种将对抗训练与似然目标结合的两阶段方法，会在多大程度上错误地表示数据，仍是一个重要研究问题。
关于深度生成模型伦理考量更一般、更详细的讨论，可参见文献[13]。

## 6 结论

我们提出了潜在扩散模型，这是一种简单、高效的方法，可在不降低质量的情况下，显著提高去噪扩散模型的训练和采样效率。
基于这一方法及交叉注意力条件机制，我们的实验表明：无须任务专用架构，即可在广泛的条件图像合成任务上取得与最先进方法相比有利的结果。

**致谢。** 本工作得到德国联邦经济与能源部“KI-Absicherung——自动驾驶安全人工智能”项目，以及德国研究基金会（DFG）项目 421703927 的支持。

## 参考文献

文献条目保留原文，编号与正文一致；条目末尾的页码指原 PDF 中的引用页。

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2017, Honolulu, HI, USA, July 21-26, 2017, pages 1122–1131. IEEE Computer Society, 2017. 1

[2] Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein gan, 2017. 3

[3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019. 1, 2, 7, 8, 22, 28

[4] Holger Caesar, Jasper R. R. Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18- 22, 2018, pages 1209–1218. Computer Vision Foundation / IEEE Computer Society, 2018. 7, 20, 22

[5] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650, 2021. 9

[6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, volume 119 of Proceedings of Machine Learning Research, pages 1691–1703. PMLR, 2020. 3

[7] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR. OpenReview.net, 2021. 1

[8] Lu Chi, Borui Jiang, and Yadong Mu. Fast fourier convolution. In NeurIPS, 2020. 8

[9] Rewon Child. Very deep vaes generalize autoregressive models and can outperform them on images. CoRR, abs/2011.10650, 2020. 3

[10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019. 3

[11] Bin Dai and David P. Wipf. Diagnosing and enhancing VAE models. In ICLR (Poster). OpenReview.net, 2019. 2, 3

[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE Computer Society, 2009. 1, 5, 7, 22

[13] Emily Denton. Ethical considerations of generative ai. AI for Content Creation Workshop, CVPR, 2021. 9

[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018. 7

[15] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021. 1, 2, 3, 4, 6, 7, 8, 18, 22, 25, 26, 28

[16] Sander Dieleman. Musings on typicality, 2020. 1, 3

[17] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-toimage generation via transformers. CoRR, abs/2105.13290, 2021. 6, 7

[18] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation, 2015. 3

[19] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 1, 3

[20] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Adv. Neural Inform. Process. Syst., pages 658–666, 2016. 3

[21] Patrick Esser, Robin Rombach, Andreas Blattmann, and Bj¨orn Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. CoRR, abs/2108.08827, 2021. 6, 7, 22

[22] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. A note on data biases in generative models. arXiv preprint arXiv:2012.02516, 2020. 9

[23] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. CoRR, abs/2012.09841, 2020. 2, 3, 4, 6, 7, 21, 22, 29, 34, 36

[24] Mary Anne Franks and Ari Ezra Waldman. Sex, lies, and videotape: Deep fakes and free speech delusions. Md. L. Rev., 78:892, 2018. 9

[25] Kevin Frans, Lisa B. Soros, and Olaf Witkowski. Clipdraw: Exploring text-to-drawing synthesis through languageimage encoders. ArXiv, abs/2106.14843, 2021. 3

[26] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. CoRR, abs/2203.13131, 2022. 6, 7, 16

[27] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, 2014. 1, 2

[28] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans, 2017. 3

[29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Adv. Neural Inform. Process. Syst., pages 6626– 6637, 2017. 1, 5, 26

[30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2, 3, 4, 6, 17

[31] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. CoRR, abs/2106.15282, 2021. 1, 3, 22

[32] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 6, 7, 16, 22, 28, 37, 38

[33] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pages 5967–5976. IEEE Computer Society, 2017. 3, 4

[34] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5967–5976, 2017. 4

[35] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. H´enaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Jo˜ao Carreira. Perceiver IO: A general architecture for structured inputs &outputs. CoRR, abs/2107.14795, 2021. 4

[36] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Jo˜ao Carreira. Perceiver: General perception with iterative attention. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4651–4664. PMLR, 2021. 4, 5

[37] Manuel Jahn, Robin Rombach, and Bj¨orn Ommer. Highresolution complex scene synthesis with transformers. CoRR, abs/2105.06458, 2021. 20, 22, 27

[38] Niharika Jain, Alberto Olmo, Sailik Sengupta, Lydia Manikonda, and Subbarao Kambhampati. Imperfect imaganation: Implications of gans exacerbating biases on facial data augmentation and snapchat selfie lenses. arXiv preprint arXiv:2001.09528, 2020. 9

[39] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. CoRR, abs/1710.10196, 2017. 5, 6

[40] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401– 4410, 2019. 1

[41] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 5, 6

[42] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. CoRR, abs/1912.04958, 2019. 2, 6, 28

[43] Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon. Score matching model for unbounded data score. CoRR, abs/2106.05527, 2021. 6

[44] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, 2018. 3

[45] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. CoRR, abs/2107.00630, 2021. 1, 3, 16

[46] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR, 2014. 1, 3, 4, 29

[47] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. CoRR, abs/2106.00132, 2021. 3

[48] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In ICLR. OpenReview.net, 2021. 1

[49] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R. Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Tom Duerig, and Vittorio Ferrari. The open images dataset V4: unified image classi- fication, object detection, and visual relationship detection at scale. CoRR, abs/1811.00982, 2018. 7, 20, 22

[50] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. CoRR, abs/1904.06991, 2019. 5, 26

[51] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014. 6, 7, 27

[52] Yuqing Ma, Xianglong Liu, Shihao Bai, Le-Yi Wang, Aishan Liu, Dacheng Tao, and Edwin Hancock. Region-wise generative adversarial imageinpainting for large missing areas. ArXiv, abs/1909.12507, 2019. 9

[53] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, JunYan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. CoRR, abs/2108.01073, 2021. 1

[54] Lars M. Mescheder. On the convergence properties of GAN training. CoRR, abs/1801.04406, 2018. 3

[55] Luke Metz, Ben Poole, David Pfau, and Jascha SohlDickstein. Unrolled generative adversarial networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 3

[56] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014. 4

[57] Gautam Mittal, Jesse H. Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. CoRR, abs/2103.16091, 2021. 1

[58] Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, and Mehran Ebrahimi. Edgeconnect: Generative image inpainting with adversarial edge learning. ArXiv, abs/1901.00212, 2019. 9

[59] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. CoRR, abs/2112.10741, 2021. 6, 7, 16

[60] Anton Obukhov, Maximilian Seitzer, Po-Wei Wu, Semen Zhydenko, Jonathan Kyl, and Elvis Yu-Jing Lin. High-fidelity performance metrics for generative models in pytorch, 2020. Version: 0.3.0, DOI: 10.5281/zenodo.4957738. 26, 27

[61] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 4, 7

[62] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 22

[63] Gaurav Parmar, Dacheng Li, Kwonjoon Lee, and Zhuowen Tu. Dual contradistinctive generative autoencoder. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 823–832. Computer Vision Foundation / IEEE, 2021. 6

[64] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On buggy resizing libraries and surprising subtleties in fid calculation. arXiv preprint arXiv:2104.11222, 2021. 26

[65] David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. CoRR, abs/2104.10350, 2021. 2

[66] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. CoRR, abs/2102.12092, 2021. 1, 2, 3, 4, 7, 21, 27

[67] Ali Razavi, A¨aron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In NeurIPS, pages 14837–14847, 2019. 1, 2, 3, 22

[68] Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016. 4

[69] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the 31st International Conference on International Conference on Machine Learning, ICML, 2014. 1, 4, 29

[70] Robin Rombach, Patrick Esser, and Bj¨orn Ommer. Network-to-network translation with conditional invertible neural networks. In NeurIPS, 2020. 3

[71] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In MICCAI (3), volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015. 2, 3, 4

[72] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. CoRR, abs/2104.07636, 2021. 1, 4, 8, 16, 22, 23, 27

[73] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. CoRR, abs/1701.05517, 2017. 1, 3

[74] Dave Salvator. NVIDIA Developer Blog. https: / / developer . nvidia . com / blog / getting - immediate-speedups-with-a100-tf32, 2020. 28

[75] Robin San-Roman, Eliya Nachmani, and Lior Wolf. Noise estimation for generative diffusion models. CoRR, abs/2104.02600, 2021. 3

[76] Axel Sauer, Kashyap Chitta, Jens M¨uller, and Andreas Geiger. Projected gans converge faster. CoRR, abs/2111.01007, 2021. 6

[77] Edgar Sch¨onfeld, Bernt Schiele, and Anna Khoreva. A unet based discriminator for generative adversarial networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 8204–8213. Computer Vision Foundation / IEEE, 2020. 6

[78] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion- 400m: Open dataset of clip-filtered 400 million image-text pairs, 2021. 6, 7

[79] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, Int. Conf. Learn. Represent., 2015. 29, 43, 44, 45

[80] Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2C: diffusion-denoising models for few-shot conditional generation. CoRR, abs/2106.06819, 2021. 3

[81] Charlie Snell. Alien Dreams: An Emerging Art Scene. https : / / ml . berkeley . edu / blog / posts / clip-art/, 2021. [Online; accessed November-2021]. 2

[82] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. CoRR, abs/1503.03585, 2015. 1, 3, 4, 18

[83] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. 4

[84] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR. OpenReview.net, 2021. 3, 5, 6, 22

[85] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. CoRR, abs/2011.13456, 2020. 1, 3, 4, 18

[86] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for modern deep learning research. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 13693–13696. AAAI Press, 2020. 2

[87] Wei Sun and Tianfu Wu. Learning layout and style reconfigurable gans for controllable image synthesis. CoRR, abs/2003.11571, 2020. 22, 27

[88] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor S. Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. ArXiv, abs/2109.07161, 2021. 8, 9, 26, 32

[89] Tristan Sylvain, Pengchuan Zhang, Yoshua Bengio, R. Devon Hjelm, and Shikhar Sharma. Object-centric image generation from layouts. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 2647–2655. AAAI Press, 2021. 20, 22, 27

[90] Patrick Tinsley, Adam Czajka, and Patrick Flynn. This face does not exist... but it might be yours! identity leakage in generative models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1320–1328, 2021. 9

[91] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011, pages 1521–1528. IEEE, 2011. 9

[92] Arash Vahdat and Jan Kautz. NVAE: A deep hierarchical variational autoencoder. In NeurIPS, 2020. 3

[93] Arash Vahdat, Karsten Kreis, and Jan Kautz. Scorebased generative modeling in latent space. CoRR, abs/2106.05931, 2021. 2, 3, 5, 6

[94] Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, koray kavukcuoglu, Oriol Vinyals, and Alex Graves. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing Systems, 2016. 3

[95] A¨aron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. CoRR, abs/1601.06759, 2016. 3

[96] A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NIPS, pages 6306–6315, 2017. 2, 4, 29

[97] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 3, 4, 5, 7

[98] Rivers Have Wings. Tweet on Classifier-free guidance for autoregressive models. https://twitter.com/RiversHaveWings/status/1478093658716966912, 2022. 6

[99] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. CoRR, abs/1910.03771, 2019. 26

[100] Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. VAEBM: A symbiosis between variational autoencoders and energy-based models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 6

[101] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using VQ-VAE and transformers. CoRR, abs/2104.10157, 2021. 3

[102] Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. LSUN: construction of a large-scale image dataset using deep learning with humans in the loop. CoRR, abs/1506.03365, 2015. 5

[103] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan, 2021. 3, 4

[104] Jiahui Yu, Zhe L. Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Free-form image inpainting with gated convolution. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4470–4479, 2019. 9

[105] K. Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. ArXiv, abs/2103.14006, 2021. 23

[106] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 3, 8, 19

[107] Shengyu Zhao, Jianwei Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I-Chao Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. ArXiv, abs/2103.10428, 2021. 9

[108] Bolei Zhou, `Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40:1452–1464, 2018. 8, 9, 26

[109] Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. LAFITE: towards language-free training for text-to-image generation. CoRR, abs/2111.13792, 2021. 6, 7, 16

## 附录

![图 12：在 512² 图像上微调的语义风景模型](LDM_译文图像/figure_12.png)

**图 12：** 第 4.3.2 节语义风景模型在 $512^2$ 图像上微调后，以卷积方式生成的样本。

![图 13：文本条件高分辨率卷积采样](LDM_译文图像/figure_13.png)

**图 13：** 将无分类器扩散引导与第 4.3.2 节的卷积采样策略结合，可使用 14.5 亿参数的文本到图像模型，渲染大于其原生训练分辨率 $256^2$ 的图像。

图中提示词依次为：“毕加索画的《最后的晚餐》。”“一幅潜在空间的油画。”“黑袍甘道夫在群山中召唤雷电的史诗画作。”“山脉上方的落日，矢量图像。”

## A 版本变更记录

这里列出本版本（<https://arxiv.org/abs/2112.10752v2>）相对于前一版本（<https://arxiv.org/abs/2112.10752v1>）的变化。

- 更新了第 4.3 节文本到图像合成结果，来自一个新训练、更大的模型（14.5 亿参数）。还新增了与非常近期的竞争方法的比较，这些方法与本工作同期[59,109]或之后[26]发布于 arXiv。
- 更新了第 4.1 节、表 3 中 ImageNet 类别条件合成结果（另见 D.4），通过增大批大小重新训练模型得到。图 26、27 的相应定性结果也已更新。更新后的文本到图像模型和类别条件模型都使用无分类器引导[32]来提高视觉保真度。
- 按照 Saharia 等人[72]建议的方案进行了用户研究，为修复模型（第 4.5 节）和超分辨率模型（第 4.4 节）提供额外评估。
- 在正文中加入图 5，将图 18 移至附录，并在附录中加入图 13。

## B 去噪扩散模型的详细说明

> **阅读注释（本节符号与 DDPM 的对应）：** $q$ 表示预先指定的前向加噪过程及其推导出的后验，$p$ 表示要学习的反向生成过程；下标 0 表示干净数据，$T$ 表示终端高噪声状态。本节 $\alpha_t$ 是从 $x_0$ 到 $x_t$ 的累计信号系数。在常见 DDPM 方差保持记号下，它对应 $\sqrt{\bar\alpha_t^{\mathrm{DDPM}}}$，$\sigma_t$ 对应 $\sqrt{1-\bar\alpha_t^{\mathrm{DDPM}}}$；不要与 DDPM 中单步的 $\alpha_t^{\mathrm{DDPM}}=1-\beta_t$ 混用。迁移到潜在扩散时，将这里的 $x$ 换成 $z$ 即可。

扩散模型可以用信噪比 $\operatorname{SNR}(t)=\alpha_t^2/\sigma_t^2$ 表示，由序列 $(\alpha_t)_{t=1}^T$ 和 $(\sigma_t)_{t=1}^T$ 构成。
这两个序列从数据样本 $x_0$ 出发，定义前向扩散过程 $q$：

$$
q(x_t\mid x_0)=\mathcal N(x_t\mid\alpha_t x_0,\sigma_t^2 I),
\tag{4}
$$

> **阅读注释（式 4 与信噪比）：** 该高斯分布等价于采样 $x_t=\alpha_t x_0+\sigma_t\epsilon$，其中 $\epsilon\sim\mathcal N(0,I)$。$\alpha_t$ 控制保留多少原始信号，$\sigma_t$ 控制加入多少噪声；高 SNR 时输入较接近数据，低 SNR 时输入更接近噪声。$\alpha_{t\mid s}$ 则是从时刻 $s$ 到 $t$ 的信号系数，与从 0 到 $t$ 的 $\alpha_t$ 区分。

并且对于 $s<t$，具有以下马尔可夫结构：

$$
q(x_t\mid x_s)=\mathcal N(x_t\mid\alpha_{t\mid s}x_s,\sigma_{t\mid s}^2I),
\tag{5}
$$

$$
\alpha_{t\mid s}=\frac{\alpha_t}{\alpha_s},
\tag{6}
$$

$$
\sigma_{t\mid s}^2=\sigma_t^2-\alpha_{t\mid s}^2\sigma_s^2.
\tag{7}
$$

去噪扩散模型是生成模型 $p(x_0)$，通过类似的、沿时间反向运行的马尔可夫结构，逆转这一过程，即：

$$
p(x_0)=\int_z p(x_T)\prod_{t=1}^T p(x_{t-1}\mid x_t).
\tag{8}
$$

> **译注：** 式（8）按原文保留积分下标 $z$；这里的含义是对中间潜变量进行边缘化。

对应模型的证据下界（ELBO）可按离散时间步分解为：

$$
-\log p(x_0)\le
\operatorname{KL}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)
+\sum_{t=1}^T\mathbb E_{q(x_t\mid x_0)}
\operatorname{KL}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p(x_{t-1}\mid x_t)\right).
\tag{9}
$$

> **阅读注释（ELBO 的方向与各项作用）：** 式（9）右侧是负对数似然 $-\log p(x_0)$ 的变分上界；取负后才是对数似然的证据下界 ELBO。第一项约束终端加噪分布与起始先验相匹配，其余项约束学习的反向转移接近真实后验。最小化这个上界是在优化一个可计算的似然代理目标，并不意味着上界必然等于真实负对数似然。

先验 $p(x_T)$ 通常选为标准正态分布，因此 ELBO 的第一项仅依赖最终信噪比 $\operatorname{SNR}(T)$。
为最小化其余各项，$p(x_{t-1}\mid x_t)$ 的常见参数化方法，是采用真实后验 $q(x_{t-1}\mid x_t,x_0)$ 的形式，但用基于当前状态 $x_t$ 的估计 $x_\theta(x_t,t)$ 替换未知的 $x_0$。
这样得到[45]：

$$
p(x_{t-1}\mid x_t):=q(x_{t-1}\mid x_t,x_\theta(x_t,t))
\tag{10}
$$

$$
=\mathcal N\!\left(x_{t-1}\mid\mu_\theta(x_t,t),
\sigma_{t\mid t-1}^2\frac{\sigma_{t-1}^2}{\sigma_t^2}I\right),
\tag{11}
$$

其中均值可表示为：

$$
\mu_\theta(x_t,t)=
\frac{\alpha_{t\mid t-1}\sigma_{t-1}^2}{\sigma_t^2}x_t
+\frac{\alpha_{t-1}\sigma_{t\mid t-1}^2}{\sigma_t^2}x_\theta(x_t,t).
\tag{12}
$$

> **阅读注释（为什么后验里可以出现未知的 x₀）：** 推导 $q(x_{t-1}\mid x_t,x_0)$ 时，将 $x_0$ 作为已知条件，利用高斯性质就能得到解析后验；训练数据也确实提供 $x_0$。生成时没有真实 $x_0$，所以式（10）用网络估计 $x_\theta(x_t,t)$ 代入后验形式，获得可采样的反向转移。这是把解析加噪模型与学习得到的去噪估计连接起来的关键。

在这种情况下，ELBO 中的求和项简化为：

$$
\begin{aligned}
&\sum_{t=1}^T\mathbb E_{q(x_t\mid x_0)}
\operatorname{KL}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p(x_{t-1}\mid x_t)\right)\\
&=\sum_{t=1}^T\mathbb E_{\mathcal N(\epsilon\mid0,I)}
\frac12\left(\operatorname{SNR}(t-1)-\operatorname{SNR}(t)\right)
\left\|x_0-x_\theta(\alpha_t x_0+\sigma_t\epsilon,t)\right\|^2.
\end{aligned}
\tag{13}
$$

> **译注：** 原 PDF 式（13）左侧将 $p(x_{t-1}\mid x_t)$ 漏写为 $p(x_{t-1})$，并漏了 KL 项的一处右括号；此处按式（9）补全条件与括号。端点 $t=1$ 的重建项须按标准 ELBO 的端点约定理解，原文未单独展开。

遵循文献[30]，我们采用重参数化：

$$
\epsilon_\theta(x_t,t)=\frac{x_t-\alpha_t x_\theta(x_t,t)}{\sigma_t},
\tag{14}
$$

将重建项表达为去噪目标：

$$
\left\|x_0-x_\theta(\alpha_t x_0+\sigma_t\epsilon,t)\right\|^2
=\frac{\sigma_t^2}{\alpha_t^2}
\left\|\epsilon-\epsilon_\theta(\alpha_t x_0+\sigma_t\epsilon,t)\right\|^2,
\tag{15}
$$

再重新加权，使各项获得相同权重，即得到式（1）。

> **阅读注释（从重建误差到简单噪声损失）：** 式（14）可以改写为 $x_\theta(x_t,t)=(x_t-\sigma_t\epsilon_\theta(x_t,t))/\alpha_t$，说明在给定加噪系数时，预测噪声与估计干净样本可以互相转换。把式（15）代入式（13）后，原 ELBO 仍包含随时间变化的权重；式（1）再将各时间步等权处理。因此，这里的“重新加权”改变了训练时对噪声尺度的重视程度，不能理解为与原始 ELBO 完全相同、仅换了一种写法。

## C 图像引导机制

![图 14：图像引导改善高分辨率全局结构](LDM_译文图像/figure_14.png)

**图 14：** 对风景图像，无条件模型的卷积采样可能产生单调、全局结构不连贯的结果（见第 2 列）。使用低分辨率图像进行 $L_2$ 引导，有助于重新建立连贯的全局结构。三列标题依次为 $256^2$ 样本、引导后的 $512^2$ 卷积样本、未经引导的 $512^2$ 卷积样本。

扩散模型的一个有趣特性是，无条件模型可以在测试时接受条件[15,82,85]。
具体而言，文献[15]提出了一种算法，使用在扩散过程各个 $x_t$ 上训练的分类器 $\log p_\Phi(y\mid x_t)$，引导在 ImageNet 上训练的无条件和条件模型。
我们直接基于这一形式，引入事后图像引导。
对于方差固定、采用噪声参数化的模型，文献[15]的引导算法为：

$$
\hat\epsilon\leftarrow\epsilon_\theta(z_t,t)
+\sqrt{1-\alpha_t^2}\,\nabla_{z_t}\log p_\Phi(y\mid z_t).
\tag{16}
$$

这可以理解为用条件分布 $\log p_\Phi(y\mid z_t)$ 修正“得分” $\epsilon_\theta$ 的更新。
迄今这种情形只应用于单类别分类模型。
我们将引导分布 $p_\Phi(y\mid\mathcal T(\mathcal D(z_0(z_t))))$ 重新解释为给定目标图像 $y$ 的通用图像到图像转换任务，其中 $\mathcal T$ 可以是适用于当前任务的任意可微变换，例如恒等映射、下采样操作等。

例如，可以假设一个固定方差 $\sigma^2=1$ 的高斯引导器，从而：

$$
\log p_\Phi(y\mid z_t)=-\frac12
\left\|y-\mathcal T\!\left(\mathcal D(z_0(z_t))\right)\right\|_2^2
\tag{17}
$$

成为一个 $L_2$ 回归目标。
图 14 展示了这一形式如何成为在 $256^2$ 图像上训练的无条件模型的上采样机制：使用 $256^2$ 的无条件样本引导 $512^2$ 图像的卷积合成，而 $\mathcal T$ 是两倍双三次下采样。
基于这一动机，我们也实验了感知相似性引导，将 $L_2$ 目标替换为 LPIPS[106]指标，见第 4.4 节。

> **译注：** 式（16）的加号按原文保留。原文将噪声预测量 $\epsilon_\theta$ 称为“得分”；使用该公式时，须结合采用的得分／噪声符号约定理解，不能将其直接视为通常定义下的得分本身。

> **阅读注释（事后图像引导如何起作用）：** 对当前 $z_t$ 先估计干净潜变量，再解码、执行变换 $\mathcal T$，与目标图像 $y$ 比较。因为这些步骤可微，相似性目标的梯度能沿整条链传回 $z_t$，从而调整采样方向。这里的 $z_0(z_t)$ 是当前估计，并非已经完成整条采样链的最终样本。这种引导需要额外的解码与梯度计算，机制和开销都不同于前述只组合噪声预测的 CFG。

## D 补充结果

### D.1 为高分辨率合成选择信噪比

![图 15：潜在空间重新缩放的影响](LDM_译文图像/figure_15.png)

**图 15：** 潜在空间重新缩放对卷积采样的影响，此处为风景图像的语义合成。见第 4.3.2 节和 D.1。三列依次为 KL 正则化、不重新缩放；KL 正则化、重新缩放；VQ 正则化、不重新缩放。

如第 4.3.2 节所述，由潜在空间方差决定的信噪比，即 $\operatorname{Var}(z)/\sigma_t^2$，会显著影响卷积采样结果。
例如，直接在 KL 正则化模型的潜在空间中训练 LDM 时（见表 8），这个比值非常高，使模型在反向去噪过程的早期就确定大量语义细节。
相反，按照 G 节描述，使用潜变量各分量的标准差重新缩放潜在空间后，信噪比会降低。
图 15 展示了这一操作对语义图像合成中卷积采样的影响。
注意，VQ 正则化空间的方差接近 1，因此无须重新缩放。

> **阅读注释（潜变量尺度也是噪声设置的一部分）：** 对 $z_t=\alpha_t z_0+\sigma_t\epsilon$，若数据潜变量的方差为 $v$，按方差比衡量的有效信噪比为 $\alpha_t^2v/\sigma_t^2$。固定噪声调度时，将 $z_0$ 放大 $k$ 倍，会把这一比值放大 $k^2$ 倍。因此，即便网络和时间步设置相同，改变潜变量尺度也会改变各时刻的去噪难度与语义结构形成的时机。本文重新缩放后全局一致性改善的结果，应在这一背景下理解。

### D.2 全部第一阶段模型

表 8 给出了在 OpenImages 数据集上训练的各种自编码模型的完整列表。

**表 8：完整自编码器模型集合。** 在 OpenImages 上训练，在 ImageNet 验证集上评估。$|\mathcal Z|$ 是码本大小，$c$ 为潜在通道数；“KL”表示 KL 正则化；† 表示不含注意力的自编码器。R-FID、R-IS 为重建图像的 FID、IS。[原表](LDM_译文图像/table_08_original.png)

| f／模型 | 码本大小或正则化 | c | R-FID ↓ | R-IS ↑ | PSNR ↑ | PSIM ↓ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 VQGAN[23] | 16384 | 256 | 4.98 | — | 19.9±3.4 | 1.83±0.42 | 0.51±0.18 |
| 16 VQGAN[23] | 1024 | 256 | 7.94 | — | 19.4±3.3 | 1.98±0.43 | 0.50±0.18 |
| 8 DALL-E[66] | 8192 | — | 32.01 | — | 22.8±2.1 | 1.95±0.51 | 0.73±0.13 |
| 32 | 16384 | 16 | 31.83 | 40.40±1.07 | 17.45±2.90 | 2.58±0.48 | 0.41±0.18 |
| 16 | 16384 | 8 | 5.15 | 144.55±3.74 | 20.83±3.61 | 1.73±0.43 | 0.54±0.18 |
| 8 | 16384 | 4 | 1.14 | 201.92±3.97 | 23.07±3.99 | 1.17±0.36 | 0.65±0.16 |
| 8 | 256 | 4 | 1.49 | 194.20±3.87 | 22.35±3.81 | 1.26±0.37 | 0.62±0.16 |
| 4 | 8192 | 3 | 0.58 | 224.78±5.35 | 27.43±4.26 | 0.53±0.21 | 0.82±0.10 |
| 4† | 8192 | 3 | 1.06 | 221.94±4.58 | 25.21±4.17 | 0.72±0.26 | 0.76±0.12 |
| 4 | 256 | 3 | 0.47 | 223.81±4.58 | 26.43±4.22 | 0.62±0.24 | 0.80±0.11 |
| 2 | 2048 | 2 | 0.16 | 232.75±5.09 | 30.85±4.12 | 0.27±0.12 | 0.91±0.05 |
| 2 | 64 | 2 | 0.40 | 226.62±4.83 | 29.13±3.46 | 0.38±0.13 | 0.90±0.05 |
| 32 | KL | 64 | 2.04 | 189.53±3.68 | 22.27±3.93 | 1.41±0.40 | 0.61±0.17 |
| 32 | KL | 16 | 7.3 | 132.75±2.71 | 20.38±3.56 | 1.88±0.45 | 0.53±0.18 |
| 16 | KL | 16 | 0.87 | 210.31±3.97 | 24.08±4.22 | 1.07±0.36 | 0.68±0.15 |
| 16 | KL | 8 | 2.63 | 178.68±4.08 | 21.94±3.92 | 1.49±0.42 | 0.59±0.17 |
| 8 | KL | 4 | 0.90 | 209.90±4.92 | 24.19±4.19 | 1.02±0.35 | 0.69±0.15 |
| 4 | KL | 3 | 0.27 | 227.57±4.89 | 27.53±4.54 | 0.55±0.24 | 0.82±0.11 |
| 2 | KL | 2 | 0.086 | 232.66±5.16 | 32.47±4.19 | 0.20±0.09 | 0.93±0.04 |

### D.3 布局到图像合成

这里提供第 4.3.1 节布局到图像模型的定量评估和更多样本。
我们分别在 COCO[4] 和 OpenImages[49] 上训练模型，并将后者进一步在 COCO 上微调。
结果见表 9。
在遵循近期最先进布局到图像方法的训练与评估协议[89]时，我们的 COCO 模型达到了它们的性能。
从 OpenImages 模型微调后，则超过了这些工作。
OpenImages 模型的 FID 比 Jahn 等人[37]的结果改善了近 11。
图 16 展示了在 COCO 上微调后的更多生成样本。

![图 16：更多布局到图像样本](LDM_译文图像/figure_16.png)

**图 16：** 最佳布局到图像模型 LDM-4 的更多样本，该模型先在 OpenImages 上训练，再在 COCO 上微调。使用 100 步 DDIM、$\eta=0$。布局来自 COCO 验证集。

**表 9：COCO[4] 和 OpenImages[49] 上布局到图像模型的定量比较。** †：从头在 COCO 训练；*：从 OpenImages 微调。[原表](LDM_译文图像/table_09_original.png)

| 方法 | COCO 256×256：FID ↓ | OpenImages 256×256：FID ↓ | OpenImages 512×512：FID ↓ |
|---|---:|---:|---:|
| LostGAN-V2[87] | 42.55 | — | — |
| OC-GAN[89] | 41.65 | — | — |
| SPADE[62] | 41.11 | — | — |
| VQGAN+T[37] | 56.58 | 45.33 | 48.11 |
| LDM-8（本文，100 步） | 42.06† | — | — |
| LDM-4（本文，200 步） | 40.91* | 32.02 | 35.80 |

### D.4 ImageNet 上的类别条件图像合成

表 10 给出了以 FID 和 Inception 分数（IS）衡量的类别条件 LDM 结果。
LDM-8 以明显更少的参数量与计算需求（见表 18），取得很有竞争力的性能。
与此前工作类似，可以在每个噪声尺度训练分类器，并使用其引导以进一步提升性能，见附录 C。
与像素空间方法不同，该分类器在潜在空间中训练，成本很低。
更多定性结果见图 26、27。

**表 10：ImageNet[12] 类别条件图像生成的详细比较。** *：文献[67]提出的分类器拒绝采样，比例按原表标记为 acc. rate（接受率）。原图表注文字将其称为 rejection rate（拒绝率），两者表述不一致，此处保留行内的接受率写法。[原表](LDM_译文图像/table_10_original.png)

| 方法 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ | 参数量 | 采样与训练设置 |
|---|---:|---:|---:|---:|---:|---|
| SR3[72] | 11.30 | — | — | — | 625M | — |
| ImageBART[21] | 21.19 | — | — | — | 3.5B | — |
| ImageBART[21] | 7.44 | — | — | — | 3.5B | 接受率 0.05* |
| VQGAN+T[23] | 17.04 | 70.6±1.8 | — | — | 1.3B | — |
| VQGAN+T[23] | 5.88 | 304.8±3.6 | — | — | 1.3B | 接受率 0.05* |
| BigGAN-deep[3] | 6.95 | 203.6±2.6 | 0.87 | 0.28 | 340M | — |
| ADM[15] | 10.94 | 100.98 | 0.69 | 0.63 | 554M | 250 步 DDIM |
| ADM-G[15] | 4.59 | 186.7 | 0.82 | 0.52 | 608M | 250 步 DDIM |
| ADM-G、ADM-U[15] | 3.85 | 221.72 | 0.84 | 0.53 | 不适用 | 2×250 步 DDIM |
| CDM[31] | 4.88 | 158.71±2.26 | — | — | 不适用 | 2×100 步 DDIM |
| LDM-8（本文） | 17.41 | 72.92±2.6 | 0.65 | 0.62 | 395M | 200 步 DDIM；训练 2.9M 步；批大小 64 |
| LDM-8-G（本文） | 8.11 | 190.43±2.60 | 0.83 | 0.36 | 506M | 200 步 DDIM；分类器强度 10；训练 2.9M 步；批大小 64 |
| LDM-8（本文） | 15.51 | 79.03±1.03 | 0.65 | 0.63 | 395M | 200 步 DDIM；训练 4.8M 步；批大小 64 |
| LDM-8-G（本文） | 7.76 | 209.52±4.24 | 0.84 | 0.35 | 506M | 200 步 DDIM；分类器强度 10；训练 4.8M 步；批大小 64 |
| LDM-4（本文） | 10.56 | 103.49±1.24 | 0.71 | 0.62 | 400M | 250 步 DDIM；训练 178K 步；批大小 1200 |
| LDM-4-G（本文） | 3.95 | 178.22±2.43 | 0.81 | 0.55 | 400M | 250 步 DDIM；无条件引导[32]强度 1.25；训练 178K 步；批大小 1200 |
| LDM-4-G（本文） | 3.60 | 247.67±5.59 | 0.87 | 0.48 | 400M | 250 步 DDIM；无条件引导[32]强度 1.5；训练 178K 步；批大小 1200 |

### D.5 样本质量与 V100 日（续第 4.1 节）

![图 17：固定计算资源下的训练进度](LDM_译文图像/figure_17.png)

**图 17：** 为完整起见，我们还报告 ImageNet 类别条件 LDM 在固定 35 个 V100 日内的训练进展。结果使用 100 步 DDIM[84]、$\kappa=0$。为提高效率，FID 在 5000 个样本上计算。

在第 4.1 节对训练过程中样本质量的评估中，我们将 FID、IS 报告为训练步数的函数。
另一种方式是将这些指标表示为所用 V100 日资源的函数。
图 17 补充了这一分析，结果在定性上相似。

### D.6 超分辨率

为了更好地比较 LDM 与像素空间扩散模型，我们扩展了表 5 的分析：将训练步数相同、参数量相近的扩散模型与 LDM 比较。
结果见表 11 最后两行，表明 LDM 在实现显著更快采样的同时取得更好性能。
图 20 给出了定性比较，展示 LDM 与像素空间扩散模型的随机样本。

> **原文脚注 1：** 由于另一个扩散模型在像素空间中运行，无法使两种架构完全相同。

**表 11：ImageNet 验证集上的四倍上采样结果（256²）。** †：FID 参考特征在验证集上计算；‡：在训练集上计算。包含一个与 LDM-4 获得相同计算量的像素空间基线。最后两行相对于之前结果额外训练了 15 轮。[原表](LDM_译文图像/table_11_original.png)

| 方法 | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| 图像回归[72] | 15.2 | 121.1 | 27.9 | 0.801 |
| SR3[72] | 5.2 | 180.1 | 26.4 | 0.762 |
| LDM-4（本文，100 步） | 2.8†/4.8‡ | 166.3 | 24.4±3.8 | 0.69±0.14 |
| LDM-4（本文，50 步，引导） | 4.4†/6.4‡ | 153.7 | 25.8±3.7 | 0.74±0.12 |
| LDM-4（本文，100 步，引导） | 4.4†/6.4‡ | 154.1 | 25.7±3.7 | 0.73±0.12 |
| LDM-4（本文，100 步，额外 15 轮） | 2.6†/4.6‡ | 169.76±5.03 | 24.4±3.8 | 0.69±0.14 |
| Pixel-DM（100 步，额外 15 轮） | 5.1†/7.1‡ | 163.06±4.67 | 24.1±3.3 | 0.59±0.12 |

#### D.6.1 LDM-BSR：通过多样图像退化构建通用超分辨率模型

![图 18：通用超分辨率模型的泛化](LDM_译文图像/figure_18.png)

**图 18：** LDM-BSR 能泛化到任意输入，可作为通用上采样器，将类别条件 LDM 的生成样本（参见图 4）放大到 $1024^2$ 分辨率。相比之下，固定退化过程（第 4.4 节）会妨碍泛化。从左到右为双三次插值、LDM-SR、LDM-BSR。

为了评估 LDM-SR 的泛化能力，我们将其同时应用于类别条件 ImageNet 模型（第 4.1 节）的 LDM 生成样本，以及从互联网抓取的图像。
有趣的是，与文献[72]一样，仅使用双三次下采样条件训练的 LDM-SR，对不遵循这一预处理的图像泛化不佳。
因此，为得到适用于广泛真实图像的超分辨率模型，我们将 LDM-SR 的双三次下采样替换为文献[105]的退化流程；真实图像可能包含相机噪声、压缩伪影、模糊与插值的复杂叠加。
BSR 退化流程以随机顺序，对图像施加 JPEG 压缩噪声、相机传感器噪声、不同的下采样插值、高斯模糊核和高斯噪声。
我们发现，直接使用文献[105]的原参数会导致非常强的退化。
由于更温和的退化似乎更适合本应用，我们调整了 BSR 退化参数，修改后的流程见代码库：<https://github.com/CompVis/latent-diffusion>。
图 18 直接比较 LDM-SR 与 LDM-BSR，说明这一方法的有效性。
后者生成的图像明显比局限于固定预处理的模型更锐利，因此更适合真实应用。
LDM-BSR 在 LSUN-Cows 上的更多结果见图 19。

> **阅读注释（为什么只用双三次退化训练会受限）：** 超分辨率学习的是与训练退化过程对应的条件分布 $p(x_{\mathrm{HR}}\mid y_{\mathrm{LR}})$。如果训练输入总是理想双三次下采样图，而实际输入含压缩、噪声和模糊，条件分布就发生了变化。多样化退化使训练覆盖更多输入情形，但恢复的纹理仍是模型生成的合理解释，不能据此认定为被测图像原有的精确细节。

## E 实现细节与超参数

### E.1 超参数

表 12、13、14、15 汇总了全部已训练 LDM 的超参数。

**表 12：产生表 1 结果的无条件 LDM 超参数。** 所有模型均在单张 NVIDIA A100 上训练。[原表](LDM_译文图像/table_12_original.png)

| 超参数 | CelebA-HQ 256×256 | FFHQ 256×256 | LSUN-Churches 256×256 | LSUN-Bedrooms 256×256 |
|---|---|---|---|---|
| f | 4 | 4 | 8 | 4 |
| z 形状 | 64×64×3 | 64×64×3 | — | 64×64×3 |
| 码本大小 | 8192 | 8192 | — | 8192 |
| 扩散步数 | 1000 | 1000 | 1000 | 1000 |
| 噪声调度 | 线性 | 线性 | 线性 | 线性 |
| 参数量 | 274M | 274M | 294M | 274M |
| 基础通道数 | 224 | 224 | 192 | 224 |
| 深度 | 2 | 2 | 2 | 2 |
| 通道倍数 | 1,2,3,4 | 1,2,3,4 | 1,2,2,4,4 | 1,2,3,4 |
| 注意力分辨率 | 32,16,8 | 32,16,8 | 32,16,8,4 | 32,16,8 |
| 每头通道数 | 32 | 32 | 24 | 32 |
| 批大小 | 48 | 42 | 96 | 48 |
| 迭代次数* | 410k | 635k | 500k | 1.9M |
| 学习率 | 9.6e-5 | 8.4e-5 | 5.e-5 | 9.6e-5 |

**表 13：第 4.1 节分析所用 ImageNet 条件 LDM 的超参数。** 所有模型在单张 NVIDIA A100 上训练。CA 表示交叉注意力。[原表](LDM_译文图像/table_13_original.png)

| 超参数 | LDM-1 | LDM-2 | LDM-4 | LDM-8 | LDM-16 | LDM-32 |
|---|---|---|---|---|---|---|
| z 形状 | 256×256×3 | 128×128×2 | 64×64×3 | 32×32×4 | 16×16×8 | 88×8×32† |
| 码本大小 | — | 2048 | 8192 | 16384 | 16384 | 16384 |
| 扩散步数 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 噪声调度 | 线性 | 线性 | 线性 | 线性 | 线性 | 线性 |
| 模型大小 | 396M | 391M | 391M | 395M | 395M | 395M |
| 基础通道数 | 192 | 192 | 192 | 256 | 256 | 256 |
| 深度 | 2 | 2 | 2 | 2 | 2 | 2 |
| 通道倍数 | 1,1,2,2,4,4 | 1,2,2,4,4 | 1,2,3,5 | 1,2,4 | 1,2,4 | 1,2,4 |
| 注意力头数 | 1 | 1 | 1 | 1 | 1 | 1 |
| 批大小 | 7 | 9 | 40 | 64 | 112 | 112 |
| 迭代次数 | 2M | 2M | 2M | 2M | 2M | 2M |
| 学习率 | 4.9e-5 | 6.3e-5 | 8e-5 | 6.4e-5 | 4.5e-5 | 4.5e-5 |
| 条件机制 | CA | CA | CA | CA | CA | CA |
| CA 分辨率 | 32,16,8 | 32,16,8 | 32,16,8 | 32,16,8 | 16,8,4 | 8,4,2 |
| 嵌入维度 | 512 | 512 | 512 | 512 | 512 | 512 |
| Transformer 深度 | 1 | 1 | 1 | 1 | 1 | 1 |

† **译注：** 原表 13、14 的 LDM-32 形状均印为 $88\times8\times32$，此处保留。按 $256/32=8$ 的下采样关系推断，第一个“88”应为“8”。

**表 14：图 7 分析所用 CelebA 无条件 LDM 超参数。** 所有模型在单张 NVIDIA A100 上训练。*：所有模型训练 50 万次迭代；若提前收敛，则使用最佳检查点评估所报告的 FID。[原表](LDM_译文图像/table_14_original.png)

| 超参数 | LDM-1 | LDM-2 | LDM-4 | LDM-8 | LDM-16 | LDM-32 |
|---|---|---|---|---|---|---|
| z 形状 | 256×256×3 | 128×128×2 | 64×64×3 | 32×32×4 | 16×16×8 | 88×8×32† |
| 码本大小 | — | 2048 | 8192 | 16384 | 16384 | 16384 |
| 扩散步数 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 噪声调度 | 线性 | 线性 | 线性 | 线性 | 线性 | 线性 |
| 模型大小 | 270M | 265M | 274M | 258M | 260M | 258M |
| 基础通道数 | 192 | 192 | 224 | 256 | 256 | 256 |
| 深度 | 2 | 2 | 2 | 2 | 2 | 2 |
| 通道倍数 | 1,1,2,2,4,4 | 1,2,2,4,4 | 1,2,3,4 | 1,2,4 | 1,2,4 | 1,2,4 |
| 注意力分辨率 | 32,16,8 | 32,16,8 | 32,16,8 | 32,16,8 | 16,8,4 | 8,4,2 |
| 每头通道数 | 32 | 32 | 32 | 32 | 32 | 32 |
| 批大小 | 9 | 11 | 48 | 96 | 128 | 128 |
| 迭代次数* | 500k | 500k | 500k | 500k | 500k | 500k |
| 学习率 | 9e-5 | 1.1e-4 | 9.6e-5 | 9.6e-5 | 1.3e-4 | 1.3e-4 |

**表 15：第 4 节条件 LDM 的超参数。** 原表说明：除修复模型在 8 张 V100 上训练外，所有模型均在单张 NVIDIA A100 上训练。[原表](LDM_译文图像/table_15_original.png)

| 超参数 | 文本→图像 | 布局→图像（OpenImages） | 布局→图像（COCO） | 类别标签→图像 | 超分辨率 | 图像修复 | 语义图→图像 |
|---|---|---|---|---|---|---|---|
| 数据集 | LAION | OpenImages | COCO | ImageNet | ImageNet | Places | Landscapes |
| f | 8 | 4 | 8 | 4 | 4 | 4 | 8 |
| z 形状 | 32×32×4 | 64×64×3 | 32×32×4 | 64×64×3 | 64×64×3 | 64×64×3 | 32×32×4 |
| 码本大小 | — | 8192 | 16384 | 8192 | 8192 | 8192 | 16384 |
| 扩散步数 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 噪声调度 | 线性 | 线性 | 线性 | 线性 | 线性 | 线性 | 线性 |
| 模型大小 | 1.45B | 306M | 345M | 395M | 169M | 215M | 215M |
| 基础通道数 | 320 | 128 | 192 | 192 | 160 | 128 | 128 |
| 深度 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 通道倍数 | 1,2,4,4 | 1,2,3,4 | 1,2,4 | 1,2,3,5 | 1,2,2,4 | 1,4,8 | 1,4,8 |
| 注意力头数 | 8 | 1 | 1 | 1 | 1 | 1 | 1 |
| Dropout | — | — | 0.1 | — | — | — | — |
| 批大小 | 680 | 24 | 48 | 1200 | 64 | 128 | 48 |
| 迭代次数 | 390K | 4.4M | 170K | 178K | 860K | 360K | 360K |
| 学习率 | 1.0e-4 | 4.8e-5 | 4.8e-5 | 1.0e-4 | 6.4e-5 | 1.0e-6 | 4.8e-5 |
| 条件机制 | CA | CA | CA | CA | 拼接 | 拼接 | 拼接 |
| （交叉）注意力分辨率 | 32,16,8 | 32,16,8 | 32,16,8 | 32,16,8 | — | — | — |
| 嵌入维度 | 1280 | 512 | 512 | 512 | — | — | — |
| Transformer 深度 | 1 | 3 | 2 | 1 | — | — | — |

> **译注：** 表 15 中语义图模型的 $f=8$ 与第 4.3.2 节正文所述 $f=4$ 不一致；表 15 的类别条件模型参数量为 395M，表 3、10 中为 400M。译稿分别保留各处原值。

### E.2 实现细节

#### E.2.1 条件 LDM 中 τθ 的实现

对于文本到图像与布局到图像合成实验（第 4.3.1 节），我们将条件编码器 $\tau_\theta$ 实现为不带掩码的 Transformer，处理输入 $y$ 的词元化版本，并产生输出 $\zeta:=\tau_\theta(y)$，其中 $\zeta\in\mathbb R^{M\times d_\tau}$。
更具体地，Transformer 由 $N$ 个块构成，每个块包含全局自注意力层、层归一化和逐位置 MLP，形式如下（改编自 <https://github.com/lucidrains/x-transformers>）：

$$
\zeta\leftarrow\operatorname{TokEmb}(y)+\operatorname{PosEmb}(y)
\tag{18}
$$

对 $i=1,\ldots,N$，执行：

$$
\zeta_1\leftarrow\operatorname{LayerNorm}(\zeta)
\tag{19}
$$

$$
\zeta_2\leftarrow\operatorname{MultiHeadSelfAttention}(\zeta_1)+\zeta
\tag{20}
$$

$$
\zeta_3\leftarrow\operatorname{LayerNorm}(\zeta_2)
\tag{21}
$$

$$
\zeta\leftarrow\operatorname{MLP}(\zeta_3)+\zeta_2
\tag{22}
$$

循环结束后，执行：

$$
\zeta\leftarrow\operatorname{LayerNorm}(\zeta).
\tag{23}
$$

> **译注：** 原 PDF 在这组公式末尾另印有一个没有公式内容的编号（24），此处保留其存在说明，不为其补造公式。后续自编码器目标仍使用原编号（25）。

获得 $\zeta$ 后，通过图 3 所示交叉注意力机制将条件映射至 UNet。
我们修改了“ablated UNet”[15]架构，将自注意力层替换为一个浅层、不带掩码的 Transformer；它由 $T$ 个块构成，交替包含自注意力、逐位置 MLP 和交叉注意力，见表 16。
如果去掉 MLP 与交叉注意力，这一架构便等同于“ablated UNet”。

虽然可以通过额外以时间步 $t$ 为条件来提高 $\tau_\theta$ 的表示能力，但我们没有采用这种选择，因为它会降低推理速度。
对此修改的更详细分析留待未来工作。

文本到图像模型使用公开的分词器[99]：<https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast>。
布局到图像模型将边界框的空间位置离散化，并把每个框编码成 $(l,b,c)$ 三元组，其中 $l$ 表示离散左上角位置，$b$ 表示右下角位置，$c$ 包含类别信息。
上述两个任务的 $\tau_\theta$ 超参数见表 17，UNet 超参数见表 13。
第 4.1 节的类别条件模型同样采用交叉注意力实现；其中 $\tau_\theta$ 是单个 512 维的可学习嵌入层，将类别 $y$ 映射为 $\zeta\in\mathbb R^{1\times512}$。

**表 16：替换标准“ablated UNet”[15]自注意力层的 Transformer 块架构。** 详见 E.2.1。$n_h$ 表示注意力头数，$d$ 为每个头的维度。[原表](LDM_译文图像/table_16_original.png)

| 操作 | 输出形状 |
|---|---|
| 输入 | $\mathbb R^{h\times w\times c}$ |
| LayerNorm | $\mathbb R^{h\times w\times c}$ |
| 1×1 卷积 | $\mathbb R^{h\times w\times(d\cdot n_h)}$ |
| 重塑形状 | $\mathbb R^{(h\cdot w)\times(d\cdot n_h)}$ |
| 自注意力（以下三行重复 T 次） | $\mathbb R^{(h\cdot w)\times(d\cdot n_h)}$ |
| MLP | $\mathbb R^{(h\cdot w)\times(d\cdot n_h)}$ |
| 交叉注意力 | $\mathbb R^{(h\cdot w)\times(d\cdot n_h)}$ |
| 重塑形状 | $\mathbb R^{h\times w\times(d\cdot n_h)}$ |
| 1×1 卷积 | $\mathbb R^{h\times w\times c}$ |

**表 17：第 4.3 节 Transformer 编码器实验的超参数。** [原表](LDM_译文图像/table_17_original.png)

| 超参数 | 文本→图像 | 布局→图像 |
|---|---:|---:|
| 序列长度 | 77 | 92 |
| 深度 N | 32 | 16 |
| 维度 | 1280 | 512 |

> **阅读注释（两处 Transformer 深度不要混淆）：** 表 17 的深度 $N$ 指条件编码器 $\tau_\theta$ 内的 Transformer 块数；表 15 的 Transformer 深度对应 UNet 中处理条件交互的模块。两者处于不同位置，所以文本模型这里的 32 层与表 15 的深度 1 可以同时成立。带“词元”维度的注意力输入，也不意味着整个图像生成器已经变成自回归模型。

#### E.2.2 图像修复

第 4.5 节图像修复实验使用文献[88]的代码生成合成掩码。
我们使用 Places[108] 中固定的 2000 个验证样本和 3 万个测试样本。
训练时采用 $256\times256$ 随机裁剪，评估时采用 $512\times512$ 裁剪。
这遵循文献[88]的训练与测试协议，并复现了其报告的指标（见表 7 的 †）。
图 21 补充了带注意力 LDM-4 的定性结果，图 22 补充了不带注意力第一阶段、大模型、经过微调的 LDM-4 的定性结果。

### E.3 评估细节

本节补充第 4 节实验的评估细节。

#### E.3.1 无条件与类别条件图像合成的定量结果

按照常见做法，我们基于模型生成的 5 万个样本，以及各数据集的完整训练集，估计表 1、10 中 FID、Precision、Recall[29,50]计算所需的统计量。
FID 使用 `torch-fidelity` 包[60]计算。
不过，不同数据处理流程可能导致不同结果[64]，因此我们也使用 Dhariwal 和 Nichol[15]提供的脚本评估模型。
我们发现结果大体一致，例外是 ImageNet 和 LSUN-Bedrooms：前者在 `torch-fidelity` 与 Nichol、Dhariwal 脚本下分别为 7.76 与 7.77，后者分别为 2.95 与 3.0。
我们强调，今后有必要统一样本质量评估流程。
Precision 与 Recall 同样使用 Nichol、Dhariwal 提供的脚本计算。

#### E.3.2 文本到图像合成

遵循文献[66]的评估协议，我们将表 2 中文本到图像模型的生成样本与 MS-COCO[51] 验证集的 3 万个样本比较，计算 FID 和 Inception 分数。
两项指标均使用 `torch-fidelity` 计算。

#### E.3.3 布局到图像合成

为评估表 9 中布局到图像模型在 COCO 上的样本质量，我们遵循惯例[37,87,89]，在 COCO Segmentation Challenge 划分的 2048 个未增强样本上计算 FID。
为增强可比性，我们使用与文献[37]完全相同的样本。
对于 OpenImages，同样遵循该文献协议，使用验证集中 2048 张中心裁剪的测试图像。

#### E.3.4 超分辨率

我们按照文献[72]建议的流程评估 ImageNet 超分辨率模型：在训练和评估中，都移除短边小于 256 像素的图像。
ImageNet 低分辨率图像通过带抗锯齿的双三次插值生成。
我们在验证集上生成样本，并使用 `torch-fidelity`[60]评估 FID。
此外，还使用训练集上计算的参考特征进行 FID 比较，见表 5、11。

#### E.3.5 效率分析

为提高效率，图 6、17、7 中绘制的样本质量指标基于 5000 个样本计算，因此结果可能与表 1、10 不同。
所有模型参数量相近，见表 13、14。
我们在保持训练稳定的前提下，将各模型的学习率尽量调高，因此不同实验的学习率略有差异，参见表 13、14。

> **阅读注释（比较实验数字前先对齐协议）：** FID 会受样本数、参考数据划分、预处理与特征实现影响。这里曲线使用 5000 个样本，主要结果表通常使用 5 万个样本；超分辨率表中的 †、‡ 又使用不同参考划分。因此，同名指标出现差异时，应先核对这些设置，再判断模型改进幅度；单次的小幅差异也不足以直接说明稳定优势。

#### E.3.6 用户研究

对于表 4 的用户研究结果，我们遵循文献[72]的协议，采用二选一强制选择范式，评估两项不同任务的人类偏好分数。
任务 1 中，低分辨率／被掩码图像置于中间，两侧分别为对应的真实高分辨率／完整图像，以及以中间图像为条件合成的图像。
超分辨率任务的问题为：“两张图像中，哪一张更好地体现了中间低分辨率图像的高质量版本？”
图像修复任务的问题为：“两张图像中，哪一张对中间图像的缺失区域进行了更真实的修复？”
任务 2 中，类似地向受试者展示低分辨率／被掩码版本，并让其在两个竞争方法生成的对应图像间选择偏好。
与文献[72]一样，受试者在回答前观看图像 3 秒。

## F 计算需求

表 18 更详细地分析了我们使用的计算资源，并使用近期最先进模型所报告的数据[15]，比较在 CelebA-HQ、FFHQ、LSUN 和 ImageNet 上表现最好的模型。
由于这些工作使用 V100 日报告计算量，而我们使用单张 NVIDIA A100 GPU 训练模型，因此假设 A100 相对于 V100 加速 2.2 倍[74]，将 A100 日换算成 V100 日。
这一系数对应文献[74]图 1 中 UNet 的 A100 相对 V100 加速比。
为评估样本质量，我们还报告了这些数据集上的 FID。
在显著降低所需计算资源的同时，我们接近 StyleGAN2[42]、ADM[15]等最先进方法的性能。

> **阅读注释（计算收益如何解释）：** 潜在空间降低的是多次去噪网络调用的单次成本；总推理时间还取决于采样步数、条件引导和最终解码。第一阶段训练可以跨多个任务复用，其成本会随复用次数摊薄，但比较总训练成本时仍需明确是否计入。A100 日到 V100 日的 2.2 倍换算是本文采用的近似假设，不是适用于所有架构与设置的硬件常数。

**表 18：与最先进生成模型比较训练计算需求和推理吞吐量。** 训练计算量以 V100 日计；除另有说明外，竞争方法的数据来自文献[15]。吞吐量单位为样本/秒，在单张 NVIDIA A100 上测量。†：数据来自文献[15]；‡：假设使用 2500 万个训练样本；††：相对于 ImageNet 验证集的重建 FID。[原表](LDM_译文图像/table_18_original.png)

| 数据集（均为 256²） | 方法／设置 | 生成器计算量 | 分类器计算量 | 总计算量 | 推理吞吐量 | 参数量 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSUN-Churches | StyleGAN2[42]† | 64 | — | 64 | — | 59M | 3.86 | — | — | — |
| LSUN-Churches | LDM-8（本文，100 步，410K） | 18 | — | 18 | 6.80 | 256M | 4.02 | — | 0.64 | 0.52 |
| LSUN-Bedrooms | ADM[15]†（1000 步） | 232 | — | 232 | 0.03 | 552M | 1.9 | — | 0.66 | 0.51 |
| LSUN-Bedrooms | LDM-4（本文，200 步，1.9M） | 60 | — | 55 | 1.07 | 274M | 2.95 | — | 0.66 | 0.48 |
| CelebA-HQ | LDM-4（本文，500 步，410K） | 14.4 | — | 14.4 | 0.43 | 274M | 5.11 | — | 0.72 | 0.49 |
| FFHQ | StyleGAN2[42] | 32.13‡ | — | 32.13† | — | 59M | 3.8 | — | — | — |
| FFHQ | LDM-4（本文，200 步，635K） | 26 | — | 26 | 1.07 | 274M | 4.98 | — | 0.73 | 0.50 |
| ImageNet | VQGAN-f-4（本文，第一阶段） | 29 | — | 29 | — | 55M | 0.58†† | — | — | — |
| ImageNet | VQGAN-f-8（本文，第一阶段） | 66 | — | 66 | — | 68M | 1.14†† | — | — | — |
| ImageNet | BigGAN-deep[3]† | 128–256 | — | 128–256 | — | 340M | 6.95 | 203.6±2.6 | 0.87 | 0.28 |
| ImageNet | ADM[15]†（250 步） | 916 | — | 916 | 0.12 | 554M | 10.94 | 100.98 | 0.69 | 0.63 |
| ImageNet | ADM-G[15]†（25 步） | 916 | 46 | 962 | 0.7 | 608M | 5.58 | — | 0.81 | 0.49 |
| ImageNet | ADM-G[15]†（250 步） | 916 | 46 | 962 | 0.07 | 608M | 4.59 | 186.7 | 0.82 | 0.52 |
| ImageNet | ADM-G、ADM-U[15]†（250 步） | 329 | 30 | 349 | 不适用 | 不适用 | 3.85 | 221.72 | 0.84 | 0.53 |
| ImageNet | LDM-8-G（本文，100 步，2.9M） | 79 | 12 | 91 | 1.93 | 506M | 8.11 | 190.4±2.6 | 0.83 | 0.36 |
| ImageNet | LDM-8（本文，200 步 DDIM，训练 2.9M 步，批大小 64） | 79 | — | 79 | 1.9 | 395M | 17.41 | 72.92 | 0.65 | 0.62 |
| ImageNet | LDM-4（本文，250 步 DDIM，训练 178K 步，批大小 1200） | 271 | — | 271 | 0.7 | 400M | 10.56 | 103.49±1.24 | 0.71 | 0.62 |
| ImageNet | LDM-4-G（同上，无分类器引导[32]，强度 1.25） | 271 | — | 271 | 0.4 | 400M | 3.95 | 178.22±2.43 | 0.81 | 0.55 |
| ImageNet | LDM-4-G（同上，无分类器引导[32]，强度 1.5） | 271 | — | 271 | 0.4 | 400M | 3.60 | 247.67±5.59 | 0.87 | 0.48 |

> **译注：** 表 18 原表的 LSUN-Bedrooms LDM-4 一行，生成器计算量为 60、总计算量为 55；ADM-G、ADM-U 一行的 329+30 与总计算量 349 也不一致。此处均保留原值。LSUN-Churches 模型的参数量、训练步数在表 12 与表 18 亦有差异，未擅自统一。

## G 自编码器模型的细节

我们遵循文献[23]，以对抗方式训练所有自编码器模型：优化基于图像块的判别器 $D_\psi$，区分原始图像与重建图像 $\mathcal D(\mathcal E(x))$。
为避免潜在空间的尺度任意变化，我们引入正则化损失项 $L_{\mathrm{reg}}$，将潜变量 $z$ 正则化为以零为中心、方差较小的表示。
我们研究两种正则化方法：一是类似标准 VAE[46,69]，对 $q_{\mathcal E}(z\mid x)=\mathcal N(z;\mathcal E_\mu,\mathcal E_{\sigma^2})$ 与标准正态分布 $\mathcal N(z;0,1)$ 之间的 KL 散度赋予较低权重；二是使用向量量化层，通过学习包含 $|\mathcal Z|$ 个不同原型的码本，对潜在空间正则化[96]。
为获得高保真重建，两种情形都只采用非常弱的正则化：将 KL 项乘以约 $10^{-6}$ 的权重，或选择很大的码本大小 $|\mathcal Z|$。

训练自编码模型 $(\mathcal E,\mathcal D)$ 的完整目标为：

$$
\begin{aligned}
L_{\mathrm{Autoencoder}}=
\min_{\mathcal E,\mathcal D}\max_\psi\Big(&
L_{\mathrm{rec}}(x,\mathcal D(\mathcal E(x)))
-L_{\mathrm{adv}}(\mathcal D(\mathcal E(x)))\\
&+\log D_\psi(x)+L_{\mathrm{reg}}(x;\mathcal E,\mathcal D)\Big).
\end{aligned}
\tag{25}
$$

> **阅读注释（自编码器目标与扩散目标的分工）：** 式（25）用于第一阶段：重建项保留输入内容，对抗项推动局部外观真实，正则化项约束潜在空间。式（2）、（3）用于第二阶段：学习这些潜变量的生成分布。第一阶段使用判别器，并不表示每次扩散去噪都在进行对抗训练，也不表示第二阶段需要更新该判别器。

**在潜在空间中训练扩散模型。** 在学习得到的潜在空间中训练扩散模型、学习 $p(z)$ 或 $p(z\mid y)$（第 4.3 节）时，同样区分两种情况。
对于 KL 正则化潜在空间，采样：

$$
z=\mathcal E_\mu(x)+\mathcal E_\sigma(x)\cdot\varepsilon=: \mathcal E(x),
\qquad\varepsilon\sim\mathcal N(0,1).
$$

> **阅读注释（两处高斯噪声来自不同过程）：** 上式的 $\varepsilon$ 用来从图像编码器后验中采样 $z_0$，代表第一阶段的编码不确定性；式（2）及前向扩散中的 $\epsilon$ 则是随后加到 $z_0$ 上的扩散噪声。两次采样承担不同作用。第一阶段的后验正则化与第二阶段从高斯起点生成数据潜变量的过程，也应分别理解。

重新缩放潜变量时，我们从数据的第一个批次估计分量方差：

$$
\hat\sigma^2=\frac1{bchw}\sum_{b,c,h,w}
(z_{b,c,h,w}-\hat\mu)^2,
\qquad
\hat\mu=\frac1{bchw}\sum_{b,c,h,w}z_{b,c,h,w}.
$$

缩放 $\mathcal E$ 的输出，使重新缩放后的潜变量标准差为 1，即：

$$
z\leftarrow\frac z{\hat\sigma}=\frac{\mathcal E(x)}{\hat\sigma}.
$$

> **阅读注释（这里估计的是一个整体尺度）：** 按给出的公式，均值和方差对批次、通道与空间位置一起求和，得到一个标量 $\hat\sigma$，再用它统一缩放潜变量；这与逐通道标准化或逐样本白化不同。公式中的均值用于估计方差，缩放操作本身只做除法。若扩散模型训练和采样使用缩放后的坐标，则送回原始解码器前应恢复对应尺度，使编解码约定一致。

对于 VQ 正则化潜在空间，我们提取量化层之前的 $z$，将量化操作吸收到解码器中，也就是可以将其理解为 $\mathcal D$ 的第一层。

## H 更多定性结果

最后，我们提供风景模型（图 12、23、24、25）、类别条件 ImageNet 模型（图 26、27），以及 CelebA-HQ、FFHQ 和 LSUN 无条件模型（图 28–31）的更多定性结果。
类似第 4.5 节的修复模型，我们也直接在 $512^2$ 图像上微调了第 4.3.2 节的语义风景模型，定性结果见图 12、23。
对于在相对较小数据集上训练的模型，图 32–34 还展示了模型样本在 VGG[79]特征空间中的最近邻。

> **阅读注释（最近邻图能说明什么）：** 这些图检查生成样本是否与已有数据中的图像非常相似，可辅助观察模型的泛化与记忆现象。但特征空间中的距离只反映所选表示捕获的相似性；展示若干最近邻，既不能证明训练数据从未被记忆，也不能把语义相似自动判定为复制。

![图 19：LSUN-Cows 的通用超分辨率结果](LDM_译文图像/figure_19.png)

**图 19：** LDM-BSR 能泛化到任意输入，可作为通用上采样器，将 LSUN-Cows 数据集的样本放大到 $1024^2$ 分辨率。左列为双三次插值，右列为 LDM-BSR。

![图 20：潜在空间与像素空间超分辨率的随机样本比较](LDM_译文图像/figure_20.png)

**图 20：** 比较 LDM-SR 与像素空间基线扩散模型的两个随机超分辨率样本。模型经过相同训练步数，在 ImageNet 验证集上评估。列依次为输入、真实图像、像素空间基线样本 1、像素空间基线样本 2、LDM 样本 1、LDM 样本 2。

![图 21：图像修复的多样结果](LDM_译文图像/figure_21.png)

**图 21：** 图像修复的定性结果。与文献[88]不同，我们的生成式方法能够对给定输入产生多个不同样本。列依次为输入、真实图像、LaMa[88]、LDM 样本 1、LDM 样本 2、LDM 样本 3。

![图 22：更多对象移除结果](LDM_译文图像/figure_22.png)

**图 22：** 与图 11 相同设置下的更多对象移除定性结果。各组左列为输入，右列为结果。

![图 23：微调后的风景语义合成](LDM_译文图像/figure_23.png)

**图 23：** 第 4.3.2 节语义风景模型在 $512^2$ 图像上微调后，通过卷积采样得到的样本。数据为 Flickr-Landscapes[23]。

![图 24：语义合成的分辨率泛化](LDM_译文图像/figure_24.png)

**图 24：** 在 $256^2$ 分辨率上训练的 LDM，可以在风景图像语义合成等空间条件任务中泛化到更大分辨率，见第 4.3.2 节。

![图 25：1024×384 风景语义合成](LDM_译文图像/figure_25.png)

**图 25：** 当提供语义图作为条件时，LDM 可以泛化到明显大于训练时所见的分辨率。尽管这个模型仅在 $256^2$ 输入上训练，仍可用于创建此处所示的 $1024\times384$ 高分辨率样本。数据为 Flickr-Landscapes[23]。

![图 26：ImageNet 类别条件随机样本，s=5](LDM_译文图像/figure_26.png)

**图 26：** 在 ImageNet 上训练的 LDM-4 的随机样本。使用强度 $s=5.0$ 的无分类器引导[32]，200 步 DDIM，$\eta=1.0$。

![图 27：ImageNet 类别条件随机样本，s=3](LDM_译文图像/figure_27.png)

**图 27：** 在 ImageNet 上训练的 LDM-4 的随机样本。使用强度 $s=3.0$ 的无分类器引导[32]，200 步 DDIM，$\eta=1.0$。

![图 28：CelebA-HQ 随机样本](LDM_译文图像/figure_28.png)

**图 28：** 在 CelebA-HQ 上表现最好的 LDM-4 的随机样本。使用 500 步 DDIM，$\eta=0$，FID 为 5.15。

![图 29：FFHQ 随机样本](LDM_译文图像/figure_29.png)

**图 29：** 在 FFHQ 上表现最好的 LDM-4 的随机样本。使用 200 步 DDIM，$\eta=1$，FID 为 4.98。

![图 30：LSUN-Churches 随机样本](LDM_译文图像/figure_30.png)

**图 30：** 在 LSUN-Churches 上表现最好的 LDM-8 的随机样本。使用 200 步 DDIM，$\eta=0$，FID 为 4.48。

> **译注：** 图 28、30 图注中的 FID 分别为 5.15、4.48，与表 1 的 5.11、4.02 不同；此处按原图注保留。

![图 31：LSUN-Bedrooms 随机样本](LDM_译文图像/figure_31.png)

**图 31：** 在 LSUN-Bedrooms 上表现最好的 LDM-4 的随机样本。使用 200 步 DDIM，$\eta=1$，FID 为 2.95。

![图 32：CelebA-HQ 最近邻](LDM_译文图像/figure_32.png)

**图 32：** 最佳 CelebA-HQ 模型的最近邻，在 VGG-16[79]特征空间中计算。每行最左边的样本来自模型，其余为对应的 10 个最近邻。

![图 33：FFHQ 最近邻](LDM_译文图像/figure_33.png)

**图 33：** 最佳 FFHQ 模型的最近邻，在 VGG-16[79]特征空间中计算。每行最左边的样本来自模型，其余为对应的 10 个最近邻。

![图 34：LSUN-Churches 最近邻](LDM_译文图像/figure_34.png)

**图 34：** 最佳 LSUN-Churches 模型的最近邻，在 VGG-16[79]特征空间中计算。每行最左边的样本来自模型，其余为对应的 10 个最近邻。
