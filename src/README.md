Eine Akzeptor-Handlung eines Agenten bezieht sich folgendermaßen auf die relevanten Angebotsbeobachtungen:
Die Handlungen 0 bis (maxAmountOfOffersToOneAgent - 1) sind der Index Beobachtung des angenommenen Angebots innerhalb der Beobachtungen der Angebote.
Die Handlung maxAmountOfOffersToOneAgent entspricht einem ungültigen Index und damit der Ablehung aller Angebote.
In der Regel dürfte aber nur in äußerst seltenen Fällen der Angebots-Vektor jemals voll sein.
Die Handlung Index eines leeen Angebots ist also auch die Ablehnung aller Angebote.


Eine Angebots-Handlung eines Agenten besteht aus der Kern-ID des Kerns in der Beobachtung (also von 1 bis numberOfCores). 
Die Möglichkeit, ein Angebot nicht zu machen, besteht. Die Handlung 0 kodiert das Nicht-Machen eines Angebots.
Neben der Kern-ID kann optional auch noch ein Preis gewählt werden.
Die Preis-Handlung entspricht dem Preis aus [0,...,maxPriority].