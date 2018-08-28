data{
    int<lower=0> N; 
    real Mobs[N];  //Observed data
    real Munc[N];  //Uncertainties on observed data
}
parameters {
    real mu;
    //real<lower=0> sigma;
    real<lower=0, upper=1> Q;
    //real<lower=0> sigo;
    positive_ordered[2] sigma;
    //real<lower=0> tsigo_raw;
    real Mtrue_std[N];   //Transformed true value of M
}
transformed parameters {
    real Mtrue[N];      //True value of M
    real<lower=1> tsigo; //Transformed outlier spread
    real<lower=0> tsigo_mod;
    
    for (n in 1:N){     //Transforming into true M space
        Mtrue[n] = mu + sigma[1]*Mtrue_std[n]; 
    }
    tsigo = sigma[2]/sigma[1]; //Transforming into transform space
    tsigo_mod = tsigo - 1;
}
model {
    
    mu ~ normal(-1.7, 0.3); // Mean
    sigma[1] ~ normal(0, 0.1); // Sigma inlier
    tsigo_mod ~ normal(0, 5); // Prior on ratio of sigmas
    sigma[2] ~ normal(0, 0.5);  // Sigma outlier
    Q ~ beta(2.5,2);      // Slight favour towards Q > 0.5

    Mobs ~ normal(Mtrue, Munc);  //p(D | theta, M)
    
    //p(M | theta)
    for (n in 1:N){
        target += log_mix(Q,
                    normal_lpdf(Mtrue_std[n] | 0, 1),
                    normal_lpdf(Mtrue_std[n] | 0, tsigo));
    }
}