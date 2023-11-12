*cd /Users/iankim/Desktop/qv/project1/Sep2023_design1/Routine
cd "C:\Users\Jaeyeon Kim\OneDrive\Documents\experiment_2"

/*
=================================================================
					ROUTINE WORK AFTER HPC
=================================================================
- (1) Import data, generated from HPC
- (2) Data quality check --> generate 2X2X2X2 table pi_1 pi_2 pi_10 pi_01 pi_12 pi_00
  
********THIS STEP REQUIRES RUNNING TIME*********
- (3) Do the kernel regression 
	- (3-a) regress entry dummies onto the Z space --> predict pi_1_hat, pi_2_hat
	- (3-b) for each j:E regress price onto the Z space --> predict np_price
************************************************

- (4) Robinson Invertibility Check
	- Generate Long Form Data
	- For each row, keep market_num, x_1, x_2, w_1, w_2, x_j, np_price, price, share, ms, firm, pi_1_hat, pi_2_hat, pi_10_hat, pi_01_hat, pi_12_hat
- (5) Model 1,2 and 3
- (6) Model 4
*/


*********************************************
// (1) Import data, generated from HPC
*********************************************
import delimited using df_final_shuffled.csv, clear
save df_origin, replace


*********************************************
// (2) Data quality check
*********************************************
use df_origin, clear

local vars "x_1 x_2 w_1 w_2"
foreach var in `vars'{
	gen `var'_bin = "HIGH"
	sum `var', d
	replace `var'_bin = "LOW" if `var' <= r(p50)
}

gen f1_entry_dummy = (xi_1 >= xi_1_star)
gen f2_entry_dummy = (xi_2 >= xi_2_star)

gen f1_monopoly_dummy = (xi_1 >= xi_1_star) * (xi_2 < xi_2_star)
gen f2_monopoly_dummy = (xi_1 < xi_1_star) * (xi_2 >= xi_2_star)

gen f1f2_duo_dummy = (xi_1 >= xi_1_star) * (xi_2 >= xi_2_star)
gen none_dummy = (xi_1 < xi_1_star) * (xi_2 < xi_2_star)

save df_origin_with_entry_dummy, replace

collapse (mean) f1_entry_dummy f2_entry_dummy f1_monopoly_dummy f2_monopoly_dummy f1f2_duo_dummy none_dummy, by(x_1_bin x_2_bin w_1_bin w_2_bin)
save table_DGP, replace



*********************************************
// (3) Do the kernel regression 
*********************************************
// (3-a) regress entry dummies onto the Z space --> predict pi_1_hat, pi_2_hat, pi_10_hat, pi_01_hat, pi_12_hat
use df_origin_with_entry_dummy, clear 

npregress kernel f1_entry_dummy x_1 x_2 w_1 w_2
predict pi_1_hat

npregress kernel f2_entry_dummy x_1 x_2 w_1 w_2
predict pi_2_hat

npregress kernel f1_monopoly_dummy x_1 x_2 w_1 w_2
predict pi_10_hat

npregress kernel f2_monopoly_dummy x_1 x_2 w_1 w_2
predict pi_01_hat

npregress kernel f1f2_duo_dummy x_1 x_2 w_1 w_2
predict pi_12_hat

save df_origin_with_entry_dummy_kernel_temp, replace



// (3-b) for each j:E regress price onto the Z space --> predict np_price
use df_origin_with_entry_dummy_kernel_temp, clear

preserve
keep if ms == 1 // j=1:E={1}
npregress kernel price_1 x_1 x_2 w_1 w_2
predict np_price
keep market_num np_price
save f1mono_np_price, replace
restore

preserve
keep if ms == 2 // j=2:E={2}
npregress kernel price_2 x_1 x_2 w_1 w_2
predict np_price
keep market_num np_price
save f2mono_np_price, replace
restore

preserve
keep if ms == 3 // j=1:E={1,2}
npregress kernel price_1 x_1 x_2 w_1 w_2
predict np_price
keep market_num np_price
save f1duo_np_price, replace
restore

preserve
keep if ms == 3 // j=2:E={1,2}
npregress kernel price_1 x_1 x_2 w_1 w_2
predict np_price
keep market_num np_price
save f2duo_np_price, replace
restore


*********************************************
// (4) Robinson Invertibility Check
*********************************************
// Generate Long Form Data
// For each row, keep: market_num, x_1, x_2, w_1, w_2, x_j, np_price, price, share, ms, firm, pi_1_hat, pi_2_hat, pi_10_hat, pi_01_hat, pi_12_hat

// j:E = 1:{1}
use df_origin_with_entry_dummy_kernel_temp, clear
keep if ms == 1
merge 1:1 market_num using f1mono_np_price 
drop _merge

gen x_j = x_1
gen firm = 1 
rename share_1 share
gen outer_share = 1 - share
rename price_1 price
gen pi = pi_1_hat
gen pi_counter = pi_2_hat
rename pi_10_hat pi_monopoly
rename pi_12_hat pi_duopoly

keep market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
order market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
save df_f1_mono_long, replace


// j:E = 2:{2}
use df_origin_with_entry_dummy_kernel_temp, clear
keep if ms == 2
merge 1:1 market_num using f2mono_np_price 
drop _merge

gen x_j = x_2
gen firm = 2 
rename share_2 share
gen outer_share = 1 - share
rename price_2 price
gen pi = pi_2_hat
gen pi_counter = pi_1_hat
rename pi_01_hat pi_monopoly
rename pi_12_hat pi_duopoly

keep market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
order market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
save df_f2_mono_long, replace


// j:E = 1:{1,2}
use df_origin_with_entry_dummy_kernel_temp, clear
keep if ms == 3
merge 1:1 market_num using f1duo_np_price 
drop _merge

gen x_j = x_1
gen firm = 1 
gen outer_share = 1 - share_1 - share_2
rename share_1 share
rename price_1 price
gen pi = pi_1_hat
gen pi_counter = pi_2_hat
rename pi_10_hat pi_monopoly
rename pi_12_hat pi_duopoly

keep market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
order market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
save df_f1_duo_long, replace


// j:E = 2:{2}
use df_origin_with_entry_dummy_kernel_temp, clear
keep if ms == 3
merge 1:1 market_num using f2duo_np_price 
drop _merge

gen x_j = x_2
gen firm = 2 
gen outer_share = 1 - share_1 - share_2
rename share_2 share
rename price_2 price
gen pi = pi_2_hat
gen pi_counter = pi_1_hat
rename pi_01_hat pi_monopoly
rename pi_12_hat pi_duopoly

keep market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
order market_num x_1 x_2 w_1 w_2 x_j np_price price share ms firm pi pi_counter pi_monopoly pi_duopoly outer_share pi_1_hat pi_2_hat
save df_f2_duo_long, replace


// append all the long form data
use df_f1_mono_long, clear
append using df_f2_mono_long
append using df_f1_duo_long
append using df_f2_duo_long
save df_long, replace



// Robinson Invertibility Check
use df_long, clear
npregress kernel x_j pi_1_hat pi_2_hat
predict tilde_x_j
npregress kernel np_price pi_1_hat pi_2_hat
predict tilde_np_price
save df_robinson, replace

use df_robinson, clear
gen x_j_resid = x_j - tilde_x_j
gen np_price_resid = np_price - tilde_np_price
corr x_j_resid np_price_resid, cov
display(r(Var_1)*r(Var_2) - r(cov_12)*r(cov_12))

// copy the code from the windows


*********************************************
// pi > 1, pi <0 case modify
*********************************************
use df_long, clear
local pis "pi pi_counter pi_monopoly pi_duopoly"
foreach pi_var in `pis'{
	replace `pi_var' = 0.99 if `pi_var' >= 1
	replace `pi_var' = 0.01 if `pi_var' <= 0
}
save df_long, replace


*********************************************
// (5) Model 1,2 and 3
*********************************************
use df_long, clear
gen f2_dummy = (firm==2)
gen LHS = log(share) - log(outer_share)

//model 1
ivreg LHS x_j f2_dummy (price = x_1 x_2 w_1 w_2)

//model 2
reg LHS x_j np_price i.ms f2_dummy

//model 3 
gen H_temp = invnormal(1-pi)
gen H_correction = normalden(H_temp)/pi
reg LHS x_j np_price f2_dummy H_correction



*********************************************
// (6) Model 4
*********************************************
use df_long, clear

preserve
keep if ms == 3
gen zeta_star = invnormal(1-pi)
gen zeta_star_counter = invnormal(1-pi_counter)
gen sieve_1 = normalden(zeta_star)*(1-normal(zeta_star_counter))/pi_duopoly
gen sieve_2 = normalden(zeta_star_counter)*(1-normal(zeta_star))/pi_duopoly
gen sieve_3 = normalden(zeta_star)*normalden(zeta_star_counter)/pi_duopoly
gen sieve_4 = zeta_star * sieve_3
gen sieve_5 = zeta_star_counter * sieve_4
save df_temp_duopoly, replace
restore

keep if ms != 3
gen zeta_star = invnormal(1-pi)
gen zeta_star_counter = invnormal(1-pi_counter)
gen sieve_1 = normalden(zeta_star)*normal(zeta_star_counter)/pi_monopoly
gen sieve_2 = normalden(zeta_star_counter)*(1-normal(zeta_star))/pi_monopoly
gen sieve_3 = normalden(zeta_star)*normalden(zeta_star_counter)/pi_monopoly
gen sieve_4 = zeta_star * sieve_3
gen sieve_5 = zeta_star_counter * sieve_4
append using df_temp_duopoly

gen LHS = log(share) - log(outer_share)
gen f2_dummy = (firm == 2)
gen mu_22 = (ms == 2 & firm == 2)
gen mu_13 = (ms == 3 & firm == 1)
gen mu_23 = (ms == 3 & firm == 2)
gen I_duo = (ms == 3)

local vars "I_duo"
foreach var in `vars'{
	gen `var'_sieve_1 = `var'*sieve_1
	gen `var'_sieve_2 = `var'*sieve_2
	gen `var'_sieve_3 = `var'*sieve_3
	gen `var'_sieve_4 = `var'*sieve_4
	gen `var'_sieve_5 = `var'*sieve_5
}


//model 4
reg LHS x_j np_price I_duo sieve_1 sieve_2 sieve_3 sieve_4 sieve_5 I_duo_sieve_1 I_duo_sieve_2 I_duo_sieve_3 I_duo_sieve_4 I_duo_sieve_5














