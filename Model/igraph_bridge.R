# igraph_bridge.R
suppressPackageStartupMessages(library(igraph))

args <- commandArgs(trailingOnly = TRUE)
# args: mode, edge_csv, n_nodes, method, seed, n_clusters, out_csv
mode       <- args[1]  # tune or default
edge_csv   <- args[2]
n_nodes    <- as.integer(args[3])
method     <- args[4]
seed       <- as.integer(args[5])
n_clusters <- as.integer(args[6])
out_csv    <- args[7]

set.seed(seed)

# ------------------------------------------------------------
# Helper: robust membership extraction across igraph versions
# ------------------------------------------------------------
get_membership_safe <- function(x) {
  # If already an integer/double vector of memberships
  if (is.atomic(x) && is.vector(x)) {
    return(as.integer(x))
  }
  # If it's a communities object
  return(as.integer(membership(x)))
}


count_clusters <- function(lab) {
  length(unique(lab))
}

# Binary search on resolution to approach target k
binary_search_resolution <- function(run_fun, k_target,
                                     r_lo = 1e-3, r_hi = 1e3,
                                     max_iter = 20) {
  best_lab <- NULL
  best_diff <- Inf

  for (it in seq_len(max_iter)) {
    r_mid <- sqrt(r_lo * r_hi)
    lab <- run_fun(r_mid)
    k_mid <- count_clusters(lab)
    diff <- abs(k_mid - k_target)

    if (diff < best_diff) {
      best_diff <- diff
      best_lab <- lab
    }

    if (k_mid > k_target) {
      # too many clusters -> resolution too high
      r_hi <- r_mid
    } else if (k_mid < k_target) {
      # too few clusters -> resolution too low
      r_lo <- r_mid
    } else {
      # exact match
      return(lab)
    }
  }
  return(best_lab)
}


# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------
edges <- read.csv(edge_csv, header = TRUE)

if (mode == 'default') {
    if (nrow(edges) == 0) {
      # Empty graph: each node its own community (consistent with previous behavior)
      lab <- 1:n_nodes
    } else {
      # Python -> R (0-based to 1-based)
      edges$i <- edges$i + 1
      edges$j <- edges$j + 1

      g <- graph_from_data_frame(
        d = data.frame(from = edges$i, to = edges$j, weight = edges$w),
        directed = FALSE,
        vertices = data.frame(name = 1:n_nodes)
      )

      if (is.null(E(g)$weight)) {
        E(g)$weight <- rep(1.0, ecount(g))
      }

      k <- max(2, n_clusters)

      # ------------------------------------------------------------
      # Methods with explicit k via cut_at
      # ------------------------------------------------------------
      if (method == "walktrap") {
        wt <- cluster_walktrap(g, weights = E(g)$weight)
        lab <- get_membership_safe(cut_at(wt, no = k))

      } else if (method == "fast_greedy") {
        fg <- cluster_fast_greedy(g, weights = E(g)$weight)
        lab <- get_membership_safe(cut_at(fg, no = k))

      } else if (method == "leading_eigen") {
        le <- cluster_leading_eigen(g, weights = E(g)$weight)
        lab <- get_membership_safe(cut_at(le, no = k))

      # ------------------------------------------------------------
      # Louvain / Leiden: resolution fixed to 1.0 (NO tuning)
      # ------------------------------------------------------------
      } else if (method == "louvain") {
        com <- cluster_louvain(g, weights = E(g)$weight, resolution = 1.0)
        lab <- get_membership_safe(com)

      } else if (method == "leiden") {
        com <- cluster_leiden(g, weights = E(g)$weight, objective_function = "modularity", resolution_parameter = 1.0)
        lab <- get_membership_safe(com)

      # ------------------------------------------------------------
      # Methods without k (as-is)
      # ------------------------------------------------------------
      } else if (method == "infomap") {
        com <- cluster_infomap(g, e.weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else if (method == "label_prop") {
        com <- cluster_label_prop(g, weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else if (method == "spinglass") {
        com <- cluster_spinglass(g, weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else {
        stop(paste("Unknown method:", method))
      }
    }
} else if (mode == 'tune') {
    if (nrow(edges) == 0) {
      lab <- 1:n_nodes
    } else {
      edges$i <- edges$i + 1
      edges$j <- edges$j + 1

      g <- graph_from_data_frame(
        d = data.frame(from = edges$i, to = edges$j, weight = edges$w),
        directed = FALSE,
        vertices = data.frame(name = 1:n_nodes)
      )

      if (is.null(E(g)$weight)) {
        E(g)$weight <- rep(1.0, ecount(g))
      }

      k <- max(2, n_clusters)

      # ------------------------------------------------------------
      # Methods with explicit k
      # ------------------------------------------------------------
      if (method == "walktrap") {
        wt <- cluster_walktrap(g, weights = E(g)$weight)
        cut <- cut_at(wt, no = k)
        lab <- get_membership_safe(cut)

      } else if (method == "fast_greedy") {
        fg <- cluster_fast_greedy(g, weights = E(g)$weight)
        cut <- cut_at(fg, no = k)
        lab <- get_membership_safe(cut)

      } else if (method == "leading_eigen") {
        le <- cluster_leading_eigen(g, weights = E(g)$weight)
        cut <- cut_at(le, no = k)
        lab <- get_membership_safe(cut)

      # ------------------------------------------------------------
      # Louvain: resolution (binary search)
      # ------------------------------------------------------------
      } else if (method == "louvain") {

        run_louvain <- function(res) {
          com <- cluster_louvain(
            g,
            weights = E(g)$weight,
            resolution = res
          )
          get_membership_safe(com)
        }

        lab <- binary_search_resolution(run_louvain, k)

      # ------------------------------------------------------------
      # Leiden: resolution (binary search)
      # ------------------------------------------------------------
      } else if (method == "leiden") {

        run_leiden <- function(res) {
          com <- cluster_leiden(
            g,
            weights = E(g)$weight,
            objective_function = "modularity",
            resolution_parameter = res
          )
          get_membership_safe(com)
        }

        lab <- binary_search_resolution(run_leiden, k)

      # ------------------------------------------------------------
      # Methods WITHOUT k / resolution (leave as-is)
      # ------------------------------------------------------------
      } else if (method == "infomap") {
        com <- cluster_infomap(g, e.weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else if (method == "label_prop") {
        com <- cluster_label_prop(g, weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else if (method == "spinglass") {
        com <- cluster_spinglass(g, weights = E(g)$weight)
        lab <- get_membership_safe(com)

      } else {
        stop(paste("Unknown method:", method))
      }
    }
}


# zero-based labels for Python
lab0 <- lab - 1
write.csv(data.frame(label = lab0), out_csv, row.names = FALSE)




