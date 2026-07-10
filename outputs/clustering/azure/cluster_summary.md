# Semantic SMS Clusters

## Method

- Embedding provider: `azure-openai`
- Embedding model: `text-embedding-3-small`
- Embedding dimensions: 1536
- Embedding notes: Azure embeddings API. Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT. API version: 2024-05-01-preview. Batch size: 256. Concurrency: 4.
- Provider role: GenAI embedding path for semantic clustering.
- Clustering algorithm: KMeans
- Cluster count: 8
- Random state: 42
- Silhouette score: 0.0262

## Cluster Overview

| cluster | size | ham | spam | spam_rate | representative theme |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 798 | 792 | 6 | 0.008 | personal coordination messages |
| 1 | 523 | 523 | 0 | 0.000 | personal coordination messages |
| 2 | 424 | 22 | 402 | 0.948 | promotional prize or claim spam |
| 3 | 888 | 868 | 20 | 0.023 | personal coordination messages |
| 4 | 420 | 415 | 5 | 0.012 | personal coordination messages |
| 5 | 304 | 0 | 304 | 1.000 | promotional prize or claim spam |
| 6 | 957 | 956 | 1 | 0.001 | personal coordination messages |
| 7 | 1258 | 1249 | 9 | 0.007 | mixed conversational messages |

## Cluster 0 Representatives

- `ham` row `4122`: Babe! How goes that day ? What are you up to ? I miss you already, my Love ... * loving kiss* ... I hope everything goes well.
- `ham` row `1359`: Good afternoon loverboy ! How goes you day ? Any luck come your way? I think of you, sweetie and send my love across the sea to make you smile and happy
- `ham` row `3752`: Buzz! Hey, my Love ! I think of you and hope your day goes well. Did you sleep in ? I miss you babe. I long for the moment we are together again*loving smile*
- `ham` row `3485`: Hello, my love! How goes that day ? I wish your well and fine babe and hope that you find some job prospects. I miss you, boytoy ... *a teasing kiss*
- `ham` row `5016`: Hello boytoy ! Geeee ... I'm missing you today. I like to send you a tm and remind you I'm thinking of you ... And you are loved ... *loving kiss*

## Cluster 1 Representatives

- `ham` row `4107`: Jus finish my lunch on my way home lor... I tot u dun wan 2 stay in sch today...
- `ham` row `5093`: My fri ah... Okie lor,goin 4 my drivin den go shoppin after tt...
- `ham` row `1439`: Wat so late still early mah. Or we juz go 4 dinner lor. Aiya i dunno...
- `ham` row `4179`: Ok lor then we go tog lor...
- `ham` row `765`: Nothing but we jus tot u would ask cos u ba gua... But we went mt faber yest... Yest jus went out already mah so today not going out... Jus call lor...

## Cluster 2 Representatives

- `spam` row `804`: FREE for 1st week! No1 Nokia tone 4 ur mobile every week just txt NOKIA to 8077 Get txting and tell ur mates. www.getzed.co.uk POBox 36504 W45WQ 16+ norm150p/tone
- `spam` row `5142`: FREE for 1st week! No1 Nokia tone 4 ur mobile every week just txt NOKIA to 8077 Get txting and tell ur mates. www.getzed.co.uk POBox 36504 W45WQ 16+ norm150p/tone
- `spam` row `2101`: SMS SERVICES. for your inclusive text credits, pls goto www.comuk.net login= ***** unsubscribe with STOP. no extra charge. help:08700469649. PO BOX420. IP4 5WE
- `spam` row `4199`: FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W45WQ norm150p/tone 16+
- `spam` row `1018`: FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W45WQ norm150p/tone 16+

## Cluster 3 Representatives

- `ham` row `4773`: Hi..i got the money da:)
- `ham` row `2518`: Yes.i'm in office da:)
- `ham` row `1491`: Ok i juz receive..
- `ham` row `1752`: Got it..mail panren paru..
- `ham` row `4712`: Ya i knw u vl giv..its ok thanks kano..anyway enjoy wit ur family wit 1st salary..:-);-)

## Cluster 4 Representatives

- `ham` row `2386`: Sorry, I'll call later
- `ham` row `703`: Sorry, I'll call later
- `ham` row `3535`: Sorry, I'll call later
- `ham` row `445`: Sorry, I'll call later
- `ham` row `2525`: Sorry, I'll call later

## Cluster 5 Representatives

- `spam` row `1599`: URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09061790121 from land line. Claim 3030. Valid 12hrs only 150ppm
- `spam` row `2911`: URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09058094454 from land line. Claim 3030. Valid 12hrs only
- `spam` row `4760`: URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09061790121 from land line. Claim 3030. Valid 12hrs only 150ppm
- `spam` row `425`: URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09058094455 from land line. Claim 3030. Valid 12hrs only
- `spam` row `5279`: URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED. Call 09061790126 from land line. Claim 3030. Valid 12hrs only 150ppm

## Cluster 6 Representatives

- `ham` row `1102`: You busy or can I come by at some point and figure out what we're doing tomorrow
- `ham` row `2554`: Oh fine, I'll be by tonight
- `ham` row `3691`: You still coming tonight?
- `ham` row `1448`: Looks like u wil b getting a headstart im leaving here bout 2.30ish but if u r desperate for my company I could head in earlier-we were goin to meet in rummer.
- `ham` row `5275`: Hi. Hope ur day * good! Back from walk, table booked for half eight. Let me know when ur coming over.

## Cluster 7 Representatives

- `ham` row `4964`: Yup ok...
- `ham` row `2534`: Yup ok...
- `ham` row `288`: Ok..
- `ham` row `2323`: Ok...
- `ham` row `3157`: Ok...
