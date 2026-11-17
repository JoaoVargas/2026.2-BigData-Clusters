# Clusters - Métricas (TDE02)

Java 17 + Apache Spark (MLlib) sobre o dataset Forest Fires (`data/input.csv`, o mesmo do TDE01).

Para k = 2..10 calcula **silhouette** e **correlação de Pearson entre todas as colunas** (por cluster e global), comparando 4 abordagens (`original`, `kmeans_clima`, `bisecting_clima`, `gmm_clima`). A adotada é `kmeans_clima` (features de clima com log, mês cíclico e winsorização).

Saída em `output/`: `silhouette.csv`, `perfis.csv`, `correlacoes.csv`, `relatorio.txt` (abordagem adotada), `comparacao.csv` e uma subpasta por abordagem. 

## Rodar
```bash
mvn -q compile exec:java   # usa .mvn/jvm.config (--add-opens exigidos pelo Spark no JDK 17)
```
