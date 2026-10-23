# Literature actually used

本目录只收录本项目实际调用过、且能核实元数据的文献；不是 NMNAT 家族综述目录。未核实 DOI/PMID 的条目不写入。

## 约定

- `pdf/`：本地 PDF 工作副本，命名 `firstauthor_year_topic.pdf`；默认不提交 Git（版权与体积）。
- `references.bib`：与下表对应的 BibTeX，可直接用于论文或报告。
- 新增文献：先核实作者、期刊、卷期、DOI/PMID，再更新表与 BibTeX；不凭记忆补引用。
- “用途/边界”说明该文献支持或限制了哪个判断；同源结构不等于 KVP40 直接证据。
- 本目录不是 `./ars-output/`；后者只放 ARS 工作流的中间产物。

## 索引

| Key | 文献 | DOI / PMID / PMCID | PDF | 用途 / 边界 |
|---|---|---|---|---|
| `lee2017kvp40` | Lee, Li & Miller (2017), *J Bacteriol* 199(9):e00855-16 | 10.1128/JB.00855-16 / PMID 28167526 / PMC5388814 | — | KVP40 NatV 全长；直接 LC-MS + ADH 支持 NMN→NAD 合成；无稳态动力学/寡聚态数据；Nudix 域有活性、无 dead 对照。 |
| `rodionova2015nadd` | Rodionova et al. (2015), *J Biol Chem* 290(12):7693–7706 | 10.1074/jbc.M114.628016 / PMID 25631047 / PMC4367272 | — | Mtb NadD W117A；Table 2 与正文 Km 方向冲突；用于 Km≠Kd/结合-催化分离的边界，不作过渡态证据。 |
| `lowe1983stereochemistry` | Lowe & Tansley (1983), *Eur J Biochem* 132(1):117–120 | 10.1111/j.1432-1033.1983.tb07334.x / PMID 6301818 | — | [α-¹⁷O]ATP → [¹⁷O]NAD⁺ 翻转；支持直接 in-line 转移；不证明 SN2(P) 过渡态或限速步。 |
| `dangelo2000nmnat` | D'Angelo et al. (2000), *Structure* 8(9):993–1004 | 10.1016/S0969-2126(00)00190-8 / PMID 10986466 | — | 1F9A ATP+Mg²⁺；ATP/Mg 接触；无 NMN；不是三元态。 |
| `zhang2003nmnat3` | Zhang et al. (2003), *J Biol Chem* 278(15):13503–13511 | 10.1074/jbc.M300073200 / PMID 12574164 | — | 1NUS NMN+AMP-CPP；无沉积 Mg；四聚体组装，A/B 为不对称单元二聚体；K140′ 跨链；NMN 多构象/部分占据；联合占位参照，不是干净 ternary。 |
| `huang2008nadm` | Huang et al. (2008), *Structure* 16(2):196–209 | 10.1016/j.str.2007.11.017 / PMID 18275811 / PMC2258087 | — | 2QJO/2QJT NadM–Nudix；2QJO 产物 NAD/POP；2QJT 的 AMP/Mn 在 Nudix 域，不能替代 NMNAT 参照。 |

## Full citations (APA 7)

- Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio phage KVP40 encodes a functional NAD+ salvage pathway. *Journal of Bacteriology, 199*(9), e00855-16. https://doi.org/10.1128/JB.00855-16
- Rodionova, I. A., Zuccola, H. J., Sorci, L., Aleshin, A. E., Kazanov, M. D., Ma, C. T., Sergienko, E., Rubin, E. J., Locher, C. P., & Osterman, A. L. (2015). Mycobacterial nicotinate mononucleotide adenylyltransferase: Structure, mechanism, and implications for drug discovery. *Journal of Biological Chemistry, 290*(12), 7693–7706. https://doi.org/10.1074/jbc.M114.628016
- Lowe, G., & Tansley, G. (1983). The stereochemical course of nucleotidyl transfer catalysed by NAD pyrophosphorylase. *European Journal of Biochemistry, 132*(1), 117–120. https://doi.org/10.1111/j.1432-1033.1983.tb07334.x
- D'Angelo, I., Raffaelli, N., Dabusti, V., Lorenzi, T., Magni, G., & Rizzi, M. (2000). Structure of nicotinamide mononucleotide adenylyltransferase: A key enzyme in NAD+ biosynthesis. *Structure, 8*(9), 993–1004. https://doi.org/10.1016/S0969-2126(00)00190-8
- Zhang, X., Kurnasov, O. V., Karthikeyan, S., Grishin, N. V., Osterman, A. L., & Zhang, H. (2003). Structural characterization of a human cytosolic NMN/NaMN adenylyltransferase and implication in human NAD biosynthesis. *Journal of Biological Chemistry, 278*(15), 13503–13511. https://doi.org/10.1074/jbc.M300073200
- Huang, N., Sorci, L., Zhang, X., Brautigam, C. A., Li, X., Raffaelli, N., Magni, G., Grishin, N. V., Osterman, A. L., & Zhang, H. (2008). Bifunctional NMN adenylyltransferase/ADP-ribose pyrophosphatase: Structure and function in bacterial NAD metabolism. *Structure, 16*(2), 196–209. https://doi.org/10.1016/j.str.2007.11.017

## Related local data (not papers)

- KVP40 genome: `data/raw/references/AY283928.2.gb` (GenBank AY283928.2), used for the WT sequence.
- HMM profiles: `data/raw/references/PF00293.hmm`, `PF01467.hmm`; profile data, not literature.

## PDF status

本轮没有把付费墙 PDF 放入仓库。PMC 的通用 `/pdf/` 端点返回的是 HTML 页面而非 PDF；其余文献未确认开放获取。若你手上有本地副本，按上表 Key 命名为 `pdf/<key>.pdf` 放入即可；不要默认提交 Git。
