# Semantic SMS Clusters

## Method

- Embedding provider: `sklearn-svd`
- Embedding model: `tfidf-word-bigram-truncated-svd`
- Embedding dimensions: 100
- Embedding notes: Local latent semantic analysis embeddings from TF-IDF plus TruncatedSVD. This is deterministic and requires no external service.
- Provider role: Local reproducibility fallback; this is not an LLM embedding model.
- Clustering algorithm: KMeans
- Cluster count: 8
- Random state: 42
- Silhouette score: 0.0928

## Cluster Overview

| cluster | size | ham | spam | spam_rate | representative theme |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 3044 | 2929 | 115 | 0.038 | personal coordination messages |
| 1 | 214 | 207 | 7 | 0.033 | personal relationship messages |
| 2 | 583 | 51 | 532 | 0.913 | promotional prize or claim spam |
| 3 | 237 | 237 | 0 | 0.000 | personal coordination messages |
| 4 | 714 | 662 | 52 | 0.073 | personal coordination messages |
| 5 | 228 | 227 | 1 | 0.004 | mixed conversational messages |
| 6 | 274 | 238 | 36 | 0.131 | mixed conversational messages |
| 7 | 278 | 274 | 4 | 0.014 | mixed conversational messages |

## Cluster 0 Representatives

- `ham` row `305`: He is a womdarfull actor
- `ham` row `410`: Headin towards busetop
- `ham` row `452`: hanks lotsly!
- `ham` row `614`: I have many dependents
- `ham` row `3321`: Kay... Since we are out already

## Cluster 1 Representatives

- `ham` row `3240`: Good. No swimsuit allowed :)
- `spam` row `1431`: For sale - arsenal dartboard. Good condition but no doubles or trebles!
- `ham` row `5389`: NOT MUCH NO FIGHTS. IT WAS A GOOD NITE!!
- `ham` row `3067`: Sounds good, keep me posted
- `ham` row `62`: Ha ha ha good joke. Girls are situation seekers.

## Cluster 2 Representatives

- `spam` row `5191`: Our records indicate u maybe entitled to 5000 pounds in compensation for the Accident you had. To claim 4 free reply with CLAIM to this msg. 2 stop txt STOP
- `spam` row `2692`: sports fans - get the latest sports news str* 2 ur mobile 1 wk FREE PLUS a FREE TONE Txt SPORT ON to 8007 www.getzed.co.uk 0870141701216+ norm 4txt/120p
- `spam` row `2368`: Tone Club: Your subs has now expired 2 re-sub reply MONOC 4 monos or POLYC 4 polys 1 weekly @ 150p per week Txt STOP 2 stop This msg free Stream 0871212025016
- `spam` row `269`: Ur ringtone service has changed! 25 Free credits! Go to club4mobiles.com to choose content now! Stop? txt CLUB STOP to 87070. 150p/wk Club4 PO Box1146 MK45 2WT
- `spam` row `674`: Get ur 1st RINGTONE FREE NOW! Reply to this msg with TONE. Gr8 TOP 20 tones to your phone every week just £1.50 per wk 2 opt out send STOP 08452810071 16

## Cluster 3 Representatives

- `ham` row `1010`: Poyyarikatur,kolathupalayam,unjalur post,erode dis, &lt;#&gt; .
- `ham` row `201`: Found it, ENC  &lt;#&gt; , where you at?
- `ham` row `4027`: &lt;#&gt;  in mca. But not conform.
- `ham` row `2659`: Dai  &lt;#&gt;  naal eruku.
- `ham` row `3636`: Its a big difference.  &lt;#&gt;  versus  &lt;#&gt;  every  &lt;#&gt; hrs

## Cluster 4 Representatives

- `ham` row `3348`: U're welcome... Caught u using broken english again...
- `ham` row `75`: U can call me now...
- `ham` row `1554`: U too...
- `ham` row `2324`: Should I be stalking u?
- `ham` row `4284`: U can call now...

## Cluster 5 Representatives

- `ham` row `288`: Ok..
- `ham` row `1274`: Ok...
- `ham` row `1320`: Ok...
- `ham` row `1428`: Ok...
- `ham` row `1484`: Ok...

## Cluster 6 Representatives

- `ham` row `4888`: Or just do that 6times
- `ham` row `3302`: Just do what ever is easier for you
- `ham` row `5529`: Its just the effect of irritation. Just ignore it
- `ham` row `1287`: Just wondering, the others just took off
- `ham` row `270`: The evo. I just had to download flash. Jealous?

## Cluster 7 Representatives

- `ham` row `2098`: I'm done...
- `ham` row `3748`: I'm not. She lip synced with shangela.
- `ham` row `2710`: Nah, I'm a perpetual DD
- `ham` row `1483`: I'm a guy, browsin is compulsory
- `ham` row `3320`: I'm freezing and craving ice. Fml
