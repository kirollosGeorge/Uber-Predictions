Month_dummies = {
            "eleven" : 11,
            "twelve" : 12,
        }
         
Uv_dummies = {
            "zero" : 0, 
            "one" : 1 ,
            "two" : 2 ,
        }
         
Suger_dummies = {
            "one" : 1.0,
            "one_point_25" : 1.25,
            "one_half" : 1.50,
            "one_point_75" : 1.75,
            "two" : 2.0,
            "two_point_half" : 2.5,
            "three" : 3.0,
        }
car_type = {
            "UBER" : [1],
            "LYFT" : [0],
        }


icon_dummies = {
            "clear-night"         : [1,0,0,0,0,0],
            "cloudy"              : [0,1,0,0,0,0],
            "fog"                 : [0,0,1,0,0,0],
            "partly-cloudy-day"   : [0,0,0,1,0,0],
            "partly-cloudy-night" : [0,0,0,0,1,0],
            "rain"                : [0,0,0,0,0,1],
            "clear day"           : [0,0,0,0,0,0],
        }
         
name_dummies = {
            "Black SUV" :   [1,0,0,0,0,0,0,0,0,0,0,0],
            "Lux"       :   [0,1,0,0,0,0,0,0,0,0,0,0],
            "Lux Black" :   [0,0,1,0,0,0,0,0,0,0,0,0],
            "Lux Black XL": [0,0,0,1,0,0,0,0,0,0,0,0],
            "Lyft"        : [0,0,0,0,1,0,0,0,0,0,0,0],
            "Lyft XL"     : [0,0,0,0,0,1,0,0,0,0,0,0],
            "Shared"      : [0,0,0,0,0,0,1,0,0,0,0,0],
            "Taxi"        : [0,0,0,0,0,0,0,1,0,0,0,0],
            "UberPool"    : [0,0,0,0,0,0,0,0,1,0,0,0],
            "UberX"       : [0,0,0,0,0,0,0,0,0,1,0,0],
            "UberXL"      : [0,0,0,0,0,0,0,0,0,0,1,0],
            "WAV"         : [0,0,0,0,0,0,0,0,0,0,0,1],
            "Black"       : [0,0,0,0,0,0,0,0,0,0,0,0],
        }
        
product_id_dummies = {
           "6c84fd89" :   [1,0,0,0,0,0,0,0,0,0,0,0],
           "6d318bcc" :   [0,1,0,0,0,0,0,0,0,0,0,0],
           "6f72dfc5" :   [0,0,1,0,0,0,0,0,0,0,0,0],
           "8cf7e821" :   [0,0,0,1,0,0,0,0,0,0,0,0],
           "997acbb5" :   [0,0,0,0,1,0,0,0,0,0,0,0],
           "9a0e7b09" :   [0,0,0,0,0,1,0,0,0,0,0,0],
           "lyft"     :   [0,0,0,0,0,0,1,0,0,0,0,0],
           "lyft_line":   [0,0,0,0,0,0,0,1,0,0,0,0],
           "lyft_lux" :   [0,0,0,0,0,0,0,0,1,0,0,0],
           "lyft_luxsuv": [0,0,0,0,0,0,0,0,0,1,0,0],
           "lyft_plus" :  [0,0,0,0,0,0,0,0,0,0,1,0],
           "lyft_premier":[0,0,0,0,0,0,0,0,0,0,0,1],
           "55c66225"  :  [0,0,0,0,0,0,0,0,0,0,0,0],
       }
       
destination_dummies = {
           "Beacon hill"               : [1,0,0,0,0,0,0,0,0,0,0],
           "Boston University"         : [0,1,0,0,0,0,0,0,0,0,0],
           "Fen way"                   : [0,0,1,0,0,0,0,0,0,0,0],
           "Financial District"        : [0,0,0,1,0,0,0,0,0,0,0],
           "HayMarket Square"          : [0,0,0,0,1,0,0,0,0,0,0],
           "North End"                 : [0,0,0,0,0,1,0,0,0,0,0],
           "North Station"             : [0,0,0,0,0,0,1,0,0,0,0],
           "novtheastrn university"    : [0,0,0,0,0,0,0,1,0,0,0],
           "south station"             : [0,0,0,0,0,0,0,0,1,0,0],
           "theatre District"          : [0,0,0,0,0,0,0,0,0,1,0],
           "West End"                  : [0,0,0,0,0,0,0,0,0,0,1],
           "Back Bay"                  : [0,0,0,0,0,0,0,0,0,0,0],
       }

source_dummies = {
           "Beacon hill"             : [1,0,0,0,0,0,0,0,0,0,0],
           "Boston University"       : [0,1,0,0,0,0,0,0,0,0,0],
           "Fen way"                 : [0,0,1,0,0,0,0,0,0,0,0],
           "Financial District"      : [0,0,0,1,0,0,0,0,0,0,0],
           "HayMarket Square"        : [0,0,0,0,1,0,0,0,0,0,0],
           "North End"               : [0,0,0,0,0,1,0,0,0,0,0],
           "North Station"           : [0,0,0,0,0,0,1,0,0,0,0],
           "novtheastrn university"  : [0,0,0,0,0,0,0,1,0,0,0],
           "south station"           : [0,0,0,0,0,0,0,0,1,0,0],
           "theatre District"        : [0,0,0,0,0,0,0,0,0,1,0],
           "West End"                : [0,0,0,0,0,0,0,0,0,0,1],
           "Back Bay"                : [0,0,0,0,0,0,0,0,0,0,0],

       }