Cross-Validation
================

## Simulate data

``` r
nonlin_df = 
  tibble(
    id = 1:100,
    x = runif(100, 0, 1),
    y = 1 - 10 * (x - .3) ^ 2 + rnorm(100, 0, .3)
  )
```

Look at the data

``` r
nonlin_df %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point()
```

<img src="Cross_validation_files/figure-gfm/unnamed-chunk-2-1.png" width="90%" />

## Cross validation – by hand

Get training and testing datasets

``` r
train_df = sample_n(nonlin_df, size = 80)
test_df = anti_join(nonlin_df, train_df, by = "id")
```

Fit 3 models.

``` r
linear_mod = lm(y ~ x, data = train_df)
smooth_mod = gam(y ~ s(x), data = train_df)
wiggly_mod = gam(y ~ s(x), sp = 10e-6, data = train_df)
```

can I see what I just did…

``` r
train_df %>% 
  gather_predictions(linear_mod, smooth_mod, wiggly_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red") +
  facet_grid(. ~ model)
```

<img src="Cross_validation_files/figure-gfm/unnamed-chunk-5-1.png" width="90%" />
Look at prediction accuracy

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.8381133

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.3385399

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.3385399

## Cross validation using `modelr`

``` r
cv_df = 
  crossv_mc(nonlin_df, 100)
```

What is happening here…

``` r
cv_df %>% pull(train) %>% .[[1]] %>% as_tibble()
```

    ## # A tibble: 79 x 3
    ##       id      x      y
    ##    <int>  <dbl>  <dbl>
    ##  1     2 0.846  -2.05 
    ##  2     4 0.474   0.868
    ##  3     6 0.643  -0.627
    ##  4     7 0.0500  0.362
    ##  5     8 0.201   1.11 
    ##  6     9 0.500   0.392
    ##  7    10 0.106   0.844
    ##  8    11 0.229   1.05 
    ##  9    12 0.471   0.636
    ## 10    13 0.118   0.365
    ## # … with 69 more rows

``` r
cv_df %>% pull(test) %>% .[[1]] %>% as_tibble()
```

    ## # A tibble: 21 x 3
    ##       id      x      y
    ##    <int>  <dbl>  <dbl>
    ##  1     1 0.898  -2.78 
    ##  2     3 0.941  -3.11 
    ##  3     5 0.568   0.194
    ##  4    23 0.718  -0.303
    ##  5    27 0.319   1.28 
    ##  6    30 0.197   1.41 
    ##  7    32 0.189   0.531
    ##  8    33 0.602   0.411
    ##  9    35 0.0377  0.264
    ## 10    37 0.629  -0.250
    ## # … with 11 more rows

``` r
cv_df =
  cv_df %>% 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  ) %>% view()
```

Let’s try to fit models and get RMSEs for them.

``` r
cv_df %>% 
  mutate(
    linear_mod = map(.x = train, ~lm(y ~ x, data = .x))
  )
```

    ## # A tibble: 100 x 4
    ##    train             test              .id   linear_mod
    ##    <list>            <list>            <chr> <list>    
    ##  1 <tibble [79 × 3]> <tibble [21 × 3]> 001   <lm>      
    ##  2 <tibble [79 × 3]> <tibble [21 × 3]> 002   <lm>      
    ##  3 <tibble [79 × 3]> <tibble [21 × 3]> 003   <lm>      
    ##  4 <tibble [79 × 3]> <tibble [21 × 3]> 004   <lm>      
    ##  5 <tibble [79 × 3]> <tibble [21 × 3]> 005   <lm>      
    ##  6 <tibble [79 × 3]> <tibble [21 × 3]> 006   <lm>      
    ##  7 <tibble [79 × 3]> <tibble [21 × 3]> 007   <lm>      
    ##  8 <tibble [79 × 3]> <tibble [21 × 3]> 008   <lm>      
    ##  9 <tibble [79 × 3]> <tibble [21 × 3]> 009   <lm>      
    ## 10 <tibble [79 × 3]> <tibble [21 × 3]> 010   <lm>      
    ## # … with 90 more rows

Let’s try to fit models and get RMSEs for them

``` r
cv_df_new = 
cv_df %>% 
  mutate(
    linear_mod = map(.x = train, ~lm(y ~ x, data = .x)),
    smooth_mod = map(.x = train, ~gam(y ~ s(x), data = .x)),
    wiggly_mod = map(.x = train, ~gam(y ~s(x), sp = 10e-6, data = .x))
  ) %>% 
  mutate(
    rmse_linear = map2_dbl(.x = linear_mod, .y = test, ~rmse(model = .x, data = .y)),
    rmse_smooth = map2_dbl(.x = linear_mod, .y = test, ~rmse(model = .x, data = .y)),
    rmse_wiggly = map2_dbl(.x = linear_mod, .y = test, ~rmse(model = .x, data = .y))
  )
```

What do these results say about model choice?

``` r
cv_df_new %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  ggplot(aes(x = model, y = rmse)) +
  geom_violin()
```

<img src="Cross_validation_files/figure-gfm/unnamed-chunk-12-1.png" width="90%" />

Compute averages…

``` r
cv_df_new %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  group_by(model) %>% 
  summarize(avg_rmse = mean(rmse))
```

    ## `summarise()` ungrouping output (override with `.groups` argument)

    ## # A tibble: 3 x 2
    ##   model  avg_rmse
    ##   <chr>     <dbl>
    ## 1 linear    0.816
    ## 2 smooth    0.816
    ## 3 wiggly    0.816
