# --- Install Required Packages ---
install.packages(c("readr", "dplyr", "tidytext", "tm", "SnowballC", "topicmodels",
                   "ggplot2", "stringr", "textdata", "wordcloud", "RColorBrewer"))

# --- Load Libraries ---
library(readr)
library(dplyr)
library(tidytext)
library(tm)
library(SnowballC)
library(topicmodels)
library(ggplot2)
library(stringr)
library(wordcloud)
library(RColorBrewer)

# --- Step 1: Load the Data ---
file_path <- "C:/Users/USER/Documents/prothomalo_articles_with_text.csv"  # Replace with actual CSV file
data <- read_csv(file_path, show_col_types = FALSE)

# --- Step 2: Text Preprocessing ---
cat("Sample articles:\n")
print(head(data$article_text, 2))

corpus <- VCorpus(VectorSource(data$article_text))

# Define custom English stopwords (in addition to standard stopwords)
custom_stopwords <- c("said", "will", "one", "new", "can", "also", "year", 
                      "get", "just", "like", "make", "take", "us", "mr", "mps")

corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c(stopwords("en"), custom_stopwords))

cat("\nSample cleaned texts:\n")
print(sapply(corpus[1:2], as.character))

# --- Tokenization ---
text_df <- data.frame(text = sapply(corpus, as.character), stringsAsFactors = FALSE) %>%
  mutate(document = row_number())

tokens <- text_df %>%
  unnest_tokens(word, text)

cat("\nSample tokens:\n")
print(head(tokens, 20))

# --- Document-Term Matrix ---
dtm <- DocumentTermMatrix(corpus)
cat("\nDocument-Term Matrix created. Dimensions:", dim(dtm), "\n")

# Remove empty docs
rowTotals <- apply(dtm, 1, sum)
dtm <- dtm[rowTotals > 0, ]
cat("Removed empty documents. Remaining docs:", nrow(dtm), "\n")

# --- Word Frequency Analysis ---
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
word_freqs <- sort(rowSums(m), decreasing = TRUE)
freq_df <- data.frame(word = names(word_freqs), freq = word_freqs)

cat("\nTop 20 most frequent words:\n")
print(head(freq_df, 20))

# --- Set Output Directory ---
output_dir <- "C:/Users/USER/Documents/"  # Update this as needed

# --- Bar Chart ---
bar_plot <- ggplot(freq_df[1:20, ], aes(x = reorder(word, freq), y = freq)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words", x = "Word", y = "Frequency") +
  theme_minimal()
print(bar_plot)
ggsave(paste0(output_dir, "Top_20_Words_BarChart.png"), plot = bar_plot, width = 8, height = 6)

# --- Word Cloud ---
png(paste0(output_dir, "WordCloud.png"), width = 800, height = 600)
wordcloud(words = freq_df$word, freq = freq_df$freq, min.freq = 2, max.words = 100,
          random.order = FALSE, colors = brewer.pal(8, "Dark2"))
dev.off()

# --- Box Plot ---
box_plot <- ggplot(freq_df[1:50, ], aes(x = "", y = freq)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Boxplot of Top 50 Word Frequencies", y = "Frequency") +
  theme_minimal()
print(box_plot)
ggsave(paste0(output_dir, "Boxplot_Frequencies.png"), plot = box_plot, width = 6, height = 4)

# --- Line Chart ---
line_plot <- ggplot(freq_df[1:50, ], aes(x = 1:50, y = freq)) +
  geom_line(color = "blue") +
  labs(title = "Line Chart of Top 50 Word Frequencies", x = "Rank", y = "Frequency") +
  theme_minimal()
print(line_plot)
ggsave(paste0(output_dir, "LineChart_Frequencies.png"), plot = line_plot, width = 6, height = 4)

# --- Scatter Plot ---
scatter_plot <- ggplot(freq_df[1:50, ], aes(x = 1:50, y = freq)) +
  geom_point(color = "darkgreen") +
  labs(title = "Scatter Plot of Top 50 Word Frequencies", x = "Rank", y = "Frequency") +
  theme_minimal()
print(scatter_plot)
ggsave(paste0(output_dir, "ScatterPlot_Frequencies.png"), plot = scatter_plot, width = 6, height = 4)

# --- Point Plot with Labels ---
point_plot <- ggplot(freq_df[1:30, ], aes(x = reorder(word, freq), y = freq)) +
  geom_point(color = "purple", size = 3) +
  geom_text(aes(label = word), hjust = -0.2, size = 3) +
  coord_flip() +
  labs(title = "Point Plot of Top 30 Words", x = "Word", y = "Frequency") +
  theme_minimal()
print(point_plot)
ggsave(paste0(output_dir, "PointPlot_Frequencies.png"), plot = point_plot, width = 8, height = 6)

# --- Topic Modeling (LDA) ---
num_topics <- 5
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 1234))

cat("\nLDA topic modeling completed. Top terms per topic:\n")
library(tidytext)
library(tidyr)
library(tibble)

topic_terms <- tidy(lda_model, matrix = "beta")
top_terms <- topic_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)
print(top_terms)

# --- Reorder functions for ggplot ---
reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun)
}
scale_x_reordered <- function(..., sep = "___") {
  ggplot2::scale_x_discrete(labels = function(x) gsub(sep, "\n", x))
}

# --- Topic-Terms Bar Plot ---
topic_plot <- ggplot(top_terms, aes(x = reorder_within(term, beta, topic), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top Terms per Topic (LDA)", x = "Term", y = "Beta") +
  theme_minimal()
print(topic_plot)
ggsave(paste0(output_dir, "Top_Terms_Per_Topic.png"), plot = topic_plot, width = 10, height = 7)
