clear

local dataDir = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\export"
local exportDir = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\stata"
local masterDir = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification"
local genderData = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\artistData\_MASTER.dta"

local files: dir "`dataDir'" files "*.csv"


* Define label for gender dummies.
la de gender 0 "Male" 1 "Female"


* Create a .dta file from the csv files
foreach file of local files {
    clear
    cd "`dataDir'"
	local source = subinstr("`file'", ".csv", "", .)
	qui: import delimited using "`file'", varnames(1) encoding(utf-8)
	
	* Drop the unnecessary python generated variables.
	qui: capture drop prediction artistdbgender artistdbmatch exactmatch exactmatchgender nnvalidinput
	* Check the percentage of missing values
	* capture mdesc
	
	* Replace strings vars to dummies
	replace nngender = "0" if nngender == "Male"
	replace nngender = "1" if nngender == "Female"
	destring nngender, replace
	la val nngender gender
	/*
	For the neural network predictions the cutoff for accurate predictions was drawn at .2 & .8 
	Redefine by editing the following lines.
	Note: the results are in the range ]0,1[, thus .5 is the most uncertain prediction.
	*/
	gen nncutoff = abs(nnvalue - 0.5)
	replace nngender = . if nncutoff < .3
	drop nncutoff
	
	rename nngender nn_gender
	
	cd "`exportDir'"
	save "`source'.dta", replace
}

* Generate merge variables
cd "`exportDir'"
use "artimage_data-2021-04-22"
qui: gen artist = fullname
save "artimage_data-2021-04-22", replace

use "artuk-artists-2021-04-22"
qui: gen artist = forename + " " + surname if forename != "" & surname != ""
qui: replace artist = forename if surname != ""
qui: replace artist = surname if forename != ""
save "artuk-artists-2021-04-22", replace

use "artist-export-2021-04-23"
qui: gen artist = forename + " " + surname
qui: replace artist = forename if surname != ""
qui: replace artist = surname if forename != ""
save "artist-export-2021-04-23", replace


local files: dir "`exportDir'" files "*.dta"

* Merge artist database gender data
foreach file of local files {
    clear
	cd "`exportDir'"
	use "`file'"
	merge m:1 artist using "`genderData'", keepusing(gender) gen(_merge)
	* Drop if the using data did not match
	rename gender artistdb_gender
	drop if _merge == 2
	drop _merge
	drop artist
	save "`file'", replace
}


* Merge first name - gender data
local nameGenderDataLoc = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\trainingData"
local nameGenderDataRaw = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\trainingData\training_data.csv"
local nameGenderData = "C:\Users\Milan\OneDrive\Desktop\Said\GenderClassification\trainingData\nameGender.dta"

clear

* USE IF RAW DATA NOT CONVERTED
/*
import delimited using "`nameGenderDataRaw'", encoding(utf-8) varnames(1)
rename name nameMERGE
rename gender namematch_gender
drop duplicated
cd "`nameGenderDataLoc'"
save "nameGender.dta", replace
*/

local files: dir "`exportDir'" files "*.dta"

foreach file of local files {
    clear
	cd "`exportDir'"
	use "`file'"
	capture gen nameMERGE = lower(forename)
	if _rc == 111 {
	    split fullname, g("NAMES")
		rename NAMES1 nameMERGE
		drop NAMES*
	}
	merge m:1 nameMERGE using "`nameGenderData'", keepusing(namematch_gender) gen(_merge)
	* Drop if the using data did not match
	drop if _merge == 2
	drop _merge
	la val namematch_gender gender
	
	drop nameMERGE
	
	save "`file'", replace
}


* Generate final prediction variable
local files: dir "`exportDir'" files "*.dta"


foreach file of local files {
    clear
	use "`file'"
	capture drop gender inconsistency genderSource
	capture la def genderSource 0 "Already defined" 1 "Artist database" 2 "Names database" 3 "Neural Network Prediction"
	gen genderSource = .
	capture gen gender = 0 if gendername == "Male"
	if _rc == 0 {
	    replace gender = 1 if gendername == "Female"
		replace genderSource = 0 if gender != . & genderSource == .
		replace gender = artistdb_gender if gender == .
		replace genderSource = 1 if gender != . & genderSource == .
		replace gender = namematch_gender if gender == .
		replace genderSource = 2 if gender != . & genderSource == .
		replace gender = nn_gender if gender == .
		replace genderSource = 3 if gender != . & genderSource == .
		* Check for inconsistency - only for artist database!
		gen inconsistency = .
		replace inconsistency = 1 if gender != . & (gender != artistdb_gender & artistdb_gender != .)
	}
	else if _rc == 111  {
	    gen gender = artistdb_gender
		replace genderSource = 1 if gender != . & genderSource == .
		replace gender = namematch_gender if gender == .
		replace genderSource = 2 if gender != . & genderSource == .
		replace gender = nn_gender if gender == .
		replace genderSource = 3 if gender != . & genderSource == .
		* Check for inconsistency - only for artist database!
		gen inconsistency = .
		replace inconsistency = 1 if gender != . & (gender != namematch_gender & namematch_gender != .)
	}
	
	la val gender gender
	la val genderSource genderSource
	capture order gender genderSource, after(name)
	capture order gender genderSource, after(fullname)
	
	* Drop all leftover variables
	capture drop nnvalue nn_gender artistdb_gender namematch_gender
	
	save "`file'", replace
}
