# InfringeMark

A python web application to identify existing or similar trademarks based on string and phonetic similarity to prevent trademark infringement lawsuits.  

![InfringeMark](https://media.giphy.com/media/LnIdaR0EbSl02qq0YL/giphy.gif)

[You can try InfringeMark for yourself!](https://Infringemark.com)

### How it works
InfringeMark was made using historical USPTO and UK IPO court trademark court case data and uses an XGBoost model to analyze trademarks/strings across 26 features. The app finds similar trademarks that exist in the USPTO database and returns a recommendation to file or not file for a trademark.   

### NLP packages
[abydos](https://github.com/chrislit/abydos)   
[jellyfish](https://github.com/jamesturk/jellyfish)  
[rapidfuzz](https://maxbachmann.github.io/rapidfuzz/)



