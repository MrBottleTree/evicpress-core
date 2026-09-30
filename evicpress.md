EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving
ShaotingFeng∗,1 YuhanLiu∗,1 HanchenLi2 XiaokunChen3 SamuelShen3 KuntaiDu3 ZhuohanGu4
RuiZhang5 YuyangHuang1 YihuaCheng3 JiayiYao1 QizhengZhang6 GaneshAnanthanarayanan7 JunchenJiang1,3
1UniversityofChicago 2UCBerkeley 3Tensormesh,Inc. 4MIT 5UCSantaCruz 6Stanford 7Microsoft
Abstract
LRUbasedeviction
ReusingKVcacheisessentialforhighefficiencyofLarge
w/ocompression
CacheServe
LanguageModel(LLM)inferencesystems.WithmoreLLM
users,theKVcachefootprintcaneasilyexceedGPUmemory Full prefill
capacity,sopriorworkhasproposedtoeitherevictKVcache
Quality
tolower-tierstoragedevices,orcompressKVcachesothat
moreKVcachecanbefitinthefastmemory.However,prior
Fixedcompressionrate
B
workmissesanimportantopportunity:jointlyoptimizingthe + LRUbasedeviction e
tte
evictionandcompressiondecisionsacrossallKVcachesto r
minimizeaveragegenerationlatencywithouthurtingquality.
Aggressivecompression withouteviction
We propose EVICPRESS, a KV-cache management sys-
temthatapplieslossycompressionandadaptiveevictionto Time-to-first-token(TTFT)
KVcacheacrossmultiplestoragetiers.Specifically,foreach
KVcacheofacontext,EVICPRESS considerstheeffectof
Figure1:EVICPRESSjointlyoptimizesoffloadingandcom- compression and eviction of the KV cache on the average
pression,achievingmuchbettertrade-offintermsofquality
generationqualityanddelayacrossallcontextsasawhole.
andTTFT.Thisfigureisillustrative.
Toachievethis,EVICPRESSproposesaunifiedutilityfunc-
tionthatquantifiestheeffectofqualityanddelayofthelossy
compressionoreviction.Tothisend,EVICPRESS’sprofiling Realizing the potential performance benefit, however,
moduleperiodicallyupdatestheutilityfunctionscoresonall bringsanewsystemchallenge:thetotalsizeofKVcachethat
possibleeviction-compressionconfigurationsforallcontexts needstobestoredcontinuestogrowduetogrowingmodel
andplacesKVcachesusingafastheuristictorearrangeKV contextwindowlimitsandanincreasednumberofconcur-
cachesonallstoragetiers,withthegoalofmaximizingthe rentusersbeingserved.Thus,itischallengingtofitallKV
utilityfunctionscoresoneachstoragetier.Comparedtothe cacheinsideGPUmemory.Twolinesofworkexisttoavoid
baselinesthatevictKVcacheorcompressKVcache,EVIC- re-computingtheKVcachesthatdonotfitinGPUmemory.
PRESSachieveshigherKV-cachehitratesonfastdevices,i.e., First,KVcacheeviction1hasbeenproposedtoimprovethe
lowerdelay,whilepreservinghighgenerationqualitybyap- cachehitrate[6,38,59].Theseworksapplyclassiccaching
plyingconservativecompressiontocontextsthataresensitive policies,suchasleastrecentused(LRU),toevictKVcache
tocompressionerrors.Evaluationon12datasetsand5mod- fromGPUmemorytolower-tierstoragedevices,whichhave
elsdemonstratesthatEVICPRESSachievesupto2.19×faster largerandcheaperspace.Thismightincreasethecachehit
time-to-first-token(TTFT)atequivalentgenerationquality. ratebyallowingmoreKVcachestobestoredandthusavoid
repeatedly recomputing the KV cache. The second line of
1 Introduction workappliesKVcachecompressiontoreducethememory
footprintofeachKVcache,allowingmoreKVcachestofit
Large Language Models (LLMs) are used ubiquitously,in
intheGPUmemory[17,18,31,32,63].
chatbot, agentic, personal assistance, and many other use
However,bothevictionandcompressionapproachesare
cases.Thus,improvingtheefficiencyofLLMinference,which
limitedwhenusedseparately.Whencompressionisapplied,
provideslow-delayexperienceforusersatalowcomputecost,
priorworksonlyconsiderusingmoreaggressivecompression
isoneofthemostimportantproblemsinbothindustryand
whenreachingthespacelimitofthestoragedevices,anddo
academia.Tothisend,reusingKVcache—theintermediate
notconsiderthepossibilityofevictionorsimplyusingtradi-
statesgeneratedduringLLMinference—hasbecomethede-
tionalevictionpolicies,suchasLRU[17,18,31,32].When
facto optimization in modern LLM inference systems. By
eviction is applied,these works do not adapt compression
storing and reusing the KV cache of a context in different
configurationsbasedontheevictiondecisions[6,38,59].This
requests,theinferencelatencycouldbesignificantlyreduced,
andthroughputcanbegreatlyimproved[22,29,53,54]. 1KVcacheeviction,inthiswork,meansthataKVcacheismovedfroma
fasterstoragedevicetothenexttier,slower,device.Itdoesnotmeandeleting
*Equalcontribution theKVcachefromthesystem.
1
5202
ceD
61
]SO.sc[
1v64941.2152:viXra

Context 1 Context 2
canbeaggressivelycompressedtoimproveoverallTTFTand
(No quality drop (Quality collapse w/ quality.
w/ compression) any compression)
Akeyinsightisthatdifferentcontextsbenefitdifferently
4GB 8GB
fromcompressionoreviction,sincetheyhavevarioussensi-
Fast Storage Slow Storage
tivitiestolossycompression(§3.1).Somecontextsarehighly
(20GB/s) (2GB/s)
Compression Low sensitivetocompressionerrors,andevenaconservativecom-
2GB 4GB
Only Quality pressionratiocanleadtoahugequalitydrop;whileothers
Eviction canbelargelycompressedwithoutaffectinggenerationqual-
8GB 4GB High TTFT
Only
ity.Thus,compressionandevictiondecisionsshouldbemade
Compression High Quality globallytoconsiderthesensitivitiesofallcontextsinthesys-
+ Ev (O ic u ti r o s n ) 8GB 0.2 GB LowTTFT tem.Forexample,weshouldchoosetoevicthighlysensitive
contextsthatsuffergreatlyfromlossycompressionwhena
Figure2:Asimpleexampletoillustratethebenefitofjointly higherstoragetierisfull,whilebringingupKVcachefroma
decidingcompressionandeviction.ConsidertwoKVcaches lowertierthathaslowersensitivitytocompressthemaggres-
(a 4 GB KV cache that can be compressed to 5% without sivelytofitintothehigherstoragetier.
qualitydrop,anda8GBKVcachewhereslightcompression Makingglobalcachemanagementdecisionsbasedonall
dropsitsqualityto 50%),whichareoriginallystoredon a contextscreatesasearchspacetothepowerofthenumber
fastmemory(20GB/sloadingbandwidth)andaslowdevice ofcontexts.Tomakeitpractical,ourpaperproposesautility
(2GB/s). A compression-only scheme compresses both KV function,takingthestoragetier,compressionmethod,andrate
cachesbyhalf,causinglowTTFT(0.3seconds)butlowqual- asinput,outputtingasinglescorethatquantitativelymeasures
ity(only75%);aneviction-onlyschemeevictshalfoftheKV theeffectofeachdecisionondelayandquality.Thisallows
cachetoslowstorage,whichhashighquality(100%)butwith ustoeasilycomputetheoveralldelayandqualitytrade-off
highTTFT(2.4seconds)whenreusingtheevictedKVcache; basedonallcontextsforeachcachemanagementdecision.
byconsideringcompressionandeviction,wesimultaneously Putting it together, we propose EVICPRESS, a system
achievehighquality(100%)andlowTTFT(0.5seconds). that adaptively chooses the optimal compression-eviction
configuration—whethertoevict,towhichstoragedevice,and
whethertocompress,bywhatmethodandatwhichratio—on
isoftensuboptimal.AsshowninFigure2,ifweconsiderKV
cachefortwocontexts(context1is4GBinsizeandcontext2
aperKV-cache(percontext)basis.EVICPRESSfirstcalcu-
lates allpossible configurations’ utility function scores for
is8GB),context1cankeepaqualityscoreof100%evenitis
each context, based on a set of initially generated queries.
aggressivelycompressed,whilecontext2’squalitycollapses
to50%evenwithminimalcompression.Whencompression-
ThenEVICPRESSputsKVcachethatmaximizesthetotalutil-
ityfunctionscoreonacertainstoragedevice.Sincereal-world
onlyisapplied,compressingbothKVcacheto50%oftheir
originalsize,thequalityscoredegradesto50%forcontext
queriesmaydriftfromtheinitialsetofqueries,EVICPRESS
recordsnewincomingqueriesforeachcontext,andperforms
2,withanaveragequalityof75%andaverageTTFTof0.3
periodicre-profilingbasedonthenewestsetofqueries.After
seconds.Whenevictiononlyisapplied,context1isevictedto
slowdevice,andcontext2iskeptatthefastdevice,achieving
re-profiling, EVICPRESS uses a greedy-basedalgorithm to
re-arrangeKVcacheindifferentstoragetiers,suchthatthe
aqualityscoreof100%andTTFTof2.4seconds.Abetter
utilityfunctionscorescanbemaximizedoneachtier.
choiceistocombinecompressionandeviction,specifically
keepingcontext2infaststorage,andcompressingcontext1’s
WeimplementEVICPRESSwithintheexistingLLMserv-
ingstackbyextendingvLLMandLMCachewith~3Klines
KVcacheto0.2GBandevictedtoslowstorage.Thisachieves
ofcodethataddmulti-tierKV-cacheplacement,compression,
100%qualityscorewithTTFTof0.5seconds.
andeviction control. Concretely,ourimplementation inter-
Priorworksdonotconsiderdecisionsofevictionorcom-
cepts KV-cache lookups,retrievals,andstores,applies per-
pression on all contexts, primarily due to the huge search
contextcompression/evictionpolicies,andintegratestightly
spacethisproblemincurs.First,whenapplyingcompression,
withvLLM’spagedGPUmemorymanagementtoorchestrate
compression methods and rates can be adapted. Secondly,
cachemovementacrossGPU,CPU,andSSDtiers.
compressionneedstobecombinedwitheviction;forexam-
Inshort,ourcontributionsare:
ple,afteracertaincompressionrate,evictionshouldbeused
toavoidahugequalitydrop.Furthermore,whendealingwith • WedesignandimplementEVICPRESS,thefirstsystemthat
anycontext’sKVcache,consideringtheimpactofcompres- jointlyconsidersbothlossycompressionandevictionfora
sionandevictiononitselfisnotenough;instead,itsimpact multi-tierKVcacheevictionsystem.
onothercontextsalsoneedstobeconsidered.Compressinga • Weproposeautilityfunctionthatquantifiestheeffectof
KVcachetoamoderaterateandplacingitonafaststorage lossycompressionandevictiononbothqualityanddelayto
device is beneficial to its quality and TTFT; however,this makecachemanagementdecisions.Usingthisutilityfunc-
spacecanbeusedtostoresubstantiallymoreKVcachethat tion, EVICPRESS can compare all feasible compression-
2

evictionconfigurationsatoncewithaclearoptimization Dataset Length(mean) Typeofcontexts #Contexts
criterion.
|     |     |     |     |     |     |     | Samsum | 11586 | Few-shot |     | 50  |
| --- | --- | --- | --- | --- | --- | --- | ------ | ----- | -------- | --- | --- |
• WeimplementEVICPRESSinthestate-of-the-artinference
|     |     |     |     |     |     |     | TriviaQA | 12311 | Few-shot |     | 50  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | -------- | --- | --- |
engine,vLLM[24],andshowthatcomparedtobaselines
|     |     |     |     |     |     |     | MultiNews | 2741 | Multi-document |     | 50  |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---- | -------------- | --- | --- |
that apply the same compression methods and ratios on 2wikimQA 7526 Multi-document 50
all contexts, EVICPRESS reduces TTFT by 1.43–3.77× Qasper 4885 Single-document 50
across fivedifferentmodels withthe same qualityscore; NarrativeQA 40003 Single-document 50
comparedtothebaselinethatonlyappliesLRU-basedevic-
Table1:Lengths,typeandnumberofcontextsusedin§3.
tion,EVICPRESSreducesTTFTby1.22to1.56×within
| 3% quality | drop, | and improves | inference |     | throughput | by  |     |     |     |     |     |
| ---------- | ----- | ------------ | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
2.0–3.6×comparedbestbaselineataqualityscoretarget beendoneoncompressingKVcache.KVcachecompression
of80%. gives two key benefits in LLM serving. First,by reducing
thesizeofeachcacheentry,onecanstoremoreKVcaches
| We should      | note | that the idea | of        | using   | utility function |        |                |                              |     |     |            |
| -------------- | ---- | ------------- | --------- | ------- | ---------------- | ------ | -------------- | ---------------------------- | --- | --- | ---------- |
|                |      |               |           |         |                  |        | within a given | memory budget,whichincreases |     |     | the proba- |
| which captures | the  | effect of     | delay and | quality | and              | adapt- |                |                              |     |     |            |
bilityofreuseandleadstohighercachehitrates.Second,it
ingcompressionconfigurationsbasedonsensitivityarenot
new [23,39,55,62]. However,we are the firstto applythe substantiallyreducesthedatatransferoverheadinoffloading-
basedinferencesystemswhereKVcachesarestoredinCPU
| utilityfunction | to enable | the | jointoptimization |     | ofeviction |     |     |     |     |     |     |
| --------------- | --------- | --- | ----------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
orSSDmemory.ThisiscrucialsincetransferringlargeKV
| and compression,achieving |     | much | better | results | than | fixed |     |     |     |     |     |
| ------------------------- | --- | ---- | ------ | ------- | ---- | ----- | --- | --- | --- | --- | --- |
cachesoverbandwidth-limitedinterconnectscanbecomea
compressionoreviction.
performancebottleneck.
2 Background There have been manyKV cache compression methods,
andtheydifferinbothqualityretentionandsystemefficiency.
2.1 KVcachereusingandevictioninLLMs
|     |     |     |     |     |     |     | These methods   | can be broadly | categorized |          | into: (1) token   |
| --- | --- | --- | --- | --- | --- | --- | --------------- | -------------- | ----------- | -------- | ----------------- |
|     |     |     |     |     |     |     | dropping (e.g., | H2O, SnapKV)   |             | [27,59], | which selectively |
LLMinferenceincludestwophases,theprefillphasewhere
removeslessimportanttokensfromtheKVcache;(2)quanti-
theentireinputcontextisfedintotheLLMatoncetoproduce
theKVcache,anddecodephasewheretheKVcacheisused zation(e.g.,CacheGen(Adaptivequantization),KIVI(Uni-
|             |             |        |         |     |               |     | form quantization)) | [31,32],       | which | reduces | the bit-width |
| ----------- | ----------- | ------ | ------- | --- | ------------- | --- | ------------------- | -------------- | ----- | ------- | ------------- |
| to generate | consecutive | output | tokens. | The | prefill phase | is  |                     |                |       |         |               |
|             |             |        |         |     |               |     | representation      | of KV entries; | (3)   | merging | (e.g., HOMER, |
computation-heavy,especiallyunderlong-contextscenarios,
Look-M)[42,45];(4)promptcompression(e.g.,LLMlingua,
sinceitscomputationalcomplexitygrowssuper-linearlywith
LongLLMlingua)[19,20].
theinputlength.Toacceleratetheprefillphase,manyrecent
systemsimproveinferenceefficiencybystoringandreusing Duetothedifferentnaturesofthesecategories,thesemeth-
odshavewidelydifferentperformances.Aswedemonstrate
previouslycomputedKVcachestoskipredundantcomputa-
laterinSec3.2,dependingonthespecificmodelsize,dataset
tion[15,29,31].Priorworksinsystemsmainlyfocusedon
KVcachemanagementinsideahierarchicalstoragesystem. properties,andgenerationlengths,differentmethodswillhave
differenttradeoffs.Somemethods,likeCacheGenorKIVI,
Toreducememorywaste,vLLM[24]usesPagedAttention
tohandletheKVcacheinsmaller,page-likeblocksinstead haveadecompressionoverhead,whileothers,likeH2O,can
bedirectlyusedindecoding.Moreover,morenovelresearch
oflargecontiguouschunks,reducingmemoryfragmentation.
onKVcachecompressionisbeingcreatedontopofprevi-
SGLang[60]optimizescomputationandmemoryusageby
employingatree-basedcachingmechanismcalledRadixAt- ousworkeverymonth.Yetthereisnoprincipledmethodto
chooseandadoptthiswidevarietyofcompressionalgorithms
tention,whichstoresKVcachesinaradixtreeandmaintains
| themviaaleastrecentlyused(LRU)evictionpolicy.Somere- |     |     |     |     |     |     | inrealsystems. |     |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- |
centworksfurtherpushtheboundarybyextendingKVcache
|     |     |     |     |     |     |     | 3 Motivation |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
managementacrossthememoryhierarchyandstorage.LM-
Cache[29]reusesKVcachesforanyrepeatedtext,regardless ThemoststraightforwardwayofmanagingKVcachememory
ofwhetheritisaprefix,andoffloadsKVcachesthatarenot is to either apply uniform compression or to use a cache
selectedtohierarchicalcachingdevicessuchasCPUmemory. evictionscheme,suchasLeastRecentlyUsed(LRU)orLeast
WorkslikeMooncake[37],Attentionstore[13],andInstIn- FrequentlyUsed(LFU),onallcontexts.Bothcompression
fer[34]leveragedisaggregatedresources(i.e.,CPU,DRAM, andevictioncangreatlyimproveinferencethroughput,either
andSSD)tooffloadKVcachesandaccelerateLLMserving. by compressing the memory footprint of KV cache, so it
|     |     |     |     |     |     |     | is faster to  | fetch from remote | storage |      | devices [22,28,31], |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ----------------- | ------- | ---- | ------------------- |
|     |     |     |     |     |     |     | or by keeping | KV cache          | that is | more | likely to be needed |
2.2 KVcachecompression
inthenearfutureinfasterstoragedevicestomakeloading
DuetotherelativelylargesizesofKVcaches(7.81GBfor faster[5,11,13,29].
32ktokencontextinQwen3-32B),muchpreviousworkhas However,aswewillshowinthissection,furtheroptimiza-
3

tionispossible,asKVcachemanagementdecisionsshould 3.2 Why not the same compression method
dependoncontext-specificcharacteristics. andratioforallcontexts?
In this section, we present an empirical study with six
Anothernaturalquestiontoaskiswhynotapplyafixedcom-
datasetsfromLongBenchdataset[3],whichisawidely-used
pressionmethodandratioonallcontexts?Asalsodiscussed
datasetforverifying LLM’s abilityto answerlong-context
in§2.2,manyparameterscanbeadjusted,includingdiffer-
questions.ThedatasetdetailsareshowninTable1.Specif-
ent compression methods and compression ratios for each
ically,we randomly draw 50 contexts from 12 datasets in
method. As different methods look at different metrics for
theLongBench[3]dataset,coveringvarioustypesoftasks,
dropping information from the KV cache,we hypothesize
includingfew-shot,single-documentquestionanswering,and
thatdifferentcontextshavetheirownoptimalcompression
multi-documentquestionanswering.
methodandratios.
3.1 Whynotcompressorevictallcontexts?
Insight2 Theoptimalcompressionmethodandratiocannot
besimplydeterminedbythelengthsofcontextsorthetypeof
KVcachesforsomecontextsarehighlysensitivetocompres-
contexts.
sionerrors:evenapplyinganextremelyconservativecompres-
sionratiocanleadtoasubstantialqualitydrop.Incontrast,
Toverifythis,inFigure4,weplotthelengthdistributionfor
othercontextsareminimallyaffected.Thisleadstoourfirst
threegroupsofcontextswhoseoptimalcompressionmethods
insightasfollows.
are kvzip,knorm and snapkv,respectively. Here,optimal
compressionmethodisdefinedasthemethodwiththelowest
Insight1 Different contexts have different sensitivities to quality3dropatthesamecompressionratio.Thereare40.3%,
compressionerrors,thussimplyapplyingthesameeviction
40.0%and19.7%contextswhoseoptimalmethodsarekvzip,
orcompressionschemesonallcontextsisnotoptimal.
knormandsnapkv,suggestingthatthereisnodominantcom-
pressionmethodthatworksforallcontexts,andper-context
Figure 3 illustrates this,where the CDF of compression methodadaptationisneeded.Furthermore,wecanseethat
error sensitivities, defined as relative quality drop under a thecontextlengthsdistributionforthethreegroupsoverlaps
compressionratioof0.92,isplottedwithdatasetsinTable1 witheachother,indicatingthatthereisnotaclearrelationship
ofdifferenttypes.ThefigureisplottedwiththeLlama-3.1- between context length and which compression method to
8B-Instruct model with keydiff compression. As shown select.
in the figure,for example,Wikipedia articles (2wikimQA) Secondly,inFigure5,weplotthemeanoptimalcompres-
havehighsensitivities,withmedianof0.681.Ourintuition sion ratio and the standard deviation for different datasets
isthatcriticalinformationisdistributedatahighdensityin inTable1.Theoptimalcompressionratioisdefinedasthe
allcontexts.Meanwhile,storynarratives(NarrativeQA)have ratio with best quality-delay trade-off among all compres-
amuchlowersensitivity,withamedianof0.340,sincethey sionmethods4.Wecanseethatevenwithineachdataset,the
arebasedondialoguescontainingredundantinformation. optimalcompressionratiovariesgreatlyacrossdifferentcon-
Moreover,wefindthateven thoughtwodatasetsbelong texts,withcoefficientofvariationbeing0.068to0.806.This
tothesametypeofcontext,theymayhavecompletelydif- showsthattheoptimalcompressionratioscannotbesimply
ferentdistributionofsensitivities.Forexample,Samsumand determinedbythedataset.
TriviaQAarebothfew-shotdatasets,whereseveralexample Moreover,weobservethateventhoughtwodatasetsbelong
question-answerpairsareprovidedtoLLMsbeforeanswer- tothesametype,suchasMultiNewsandMusique,whichare
ingtherealquestions,whileSamsumhassensitivitywitha bothmulti-documentcontexts,theyhavenon-overlappingdis-
medianof0.676andTriviaQAhassensitivitywithamedian tributions.Thissuggeststhattheoptimalcompressionratios
of0.392.Also,MultiNewssamplesaremulti-documentcon- cannotbetriviallydeterminedbytypeofcontextseither.
texts,wheremultiplenewsarticlesareconcatenatedtoform
eachsample,andQaspersamplesaresingle-document,where
eachsampleonlycontainsasingledocumentonaspecific 4 EVICPRESS Design
topic.However,theMultiNewsandQasperdatasetshavevery
similarmediansensitivities,0.738and0.759specifically. We now describe the design of EVICPRESS, starting with
Overall, sensitivities vary significantly across contexts. anoverallworkflow(§4.1),autilityfunctionthatdetermines
Specifically,thecoefficientofvariation (ameasureofvari- thequality-ratetrade-offofeachcompressionconfiguration
ance) is from 0.078 to 0.394,and this shows that different
contextshaveverydifferentsensitivities,evenwithinthesame
3Generationqualityisdefinedasthesimilaritybetweenthegenerated
answeraftercompressionandtheoriginalprefillanswer.
dataset.
4To measure the correlation between TTFT and quality, we use
2CompressionratioisdefinedastheratiobetweencompressedKVcache normalizedquality−normalizedTTFTtomakesurethatqualityandTTFT
sizeandoriginalKVcachesize. arecomparableintermsofscale.
4

|     | 1.0 |        |     | 1.0      |     |     | 1.0 |            |     |
| --- | --- | ------ | --- | -------- | --- | --- | --- | ---------- | --- |
|     |     | samsum |     | triviaqa |     |     |     | multi_news |     |
FDC
|     | 0.5 |     |     | 0.5 |     |     | 0.5 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 0.0 |     |     | 0.0 |     |     | 0.0 |     |     |
0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00
|     | 1.0 |          |     | 1.0    |     |     | 1.0 |             |     |
| --- | --- | -------- | --- | ------ | --- | --- | --- | ----------- | --- |
|     |     | 2wikimqa |     | qasper |     |     |     | narrativeqa |     |
FDC
|     | 0.5 |     |     | 0.5 |     |     | 0.5 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 0.0 |     |     | 0.0 |     |     | 0.0 |     |     |
0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00
Relative Quality Drop Relative Quality Drop Relative Quality Drop
Figure3:DifferentcontextshavedifferentcompressionerrorsensitivitiesinthesixdatasetsmentionedinTable1.Compression
error sensitivity is defined as the quality score drop when applied with keydiff compression at a ratio of 0.9. Plotted with
Llama-3.1-8B-Instruct.
1.0
oitaR noisserpmoC lamitpO
30000
0.8
htgneL
|     | 20000 |       |       |        |     | 0.6 |     |     |     |
| --- | ----- | ----- | ----- | ------ | --- | --- | --- | --- | --- |
|     | 10000 |       |       |        |     | 0.4 |     |     |     |
|     | 0     |       |       |        |     | 0.2 |     |     |     |
|     |       | kvzip | knorm | snapkv |     |     |     |     |     |
0.0
Method
|                                                         |     |     |     |     |     | m   | a     | ws       | e er a  |
| ------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | -------- | ------- |
|                                                         |     |     |     |     |     | u   | a q   | e q u    | p e q   |
|                                                         |     |     |     |     |     | ms  | trivi | n usi    | as ativ |
|                                                         |     |     |     |     |     | a   |       | ulti _ m | q       |
|                                                         |     |     |     |     |     | s   |       |          | arr     |
| Figure4:Lengthdistributionsforcontextswhoseleastquality |     |     |     |     |     |     | m     |          | n       |
dropoccurswhenusingkvzip,knorm,andsnapkv,undera
compressionratioof0.6.
|     |     |     |     |     | Figure5: | Differenttypesofcontextshavedifferentoptimal |     |     |     |
| --- | --- | --- | --- | --- | -------- | -------------------------------------------- | --- | --- | --- |
compressionratios.Here,eachbarshowsthemeanandstan-
(§4.2),aprofilingmechanismtodeterminetherightcompres-
darddeviationofoptimalcompressionratiosfordifferenttype
| sion | configuration | for each | context (§4.3),and | a policy to | ofcontexts. |     |     |     |     |
| ---- | ------------- | -------- | ------------------ | ----------- | ----------- | --- | --- | --- | --- |
managetheKVcachewhenastoragetierisfull(§4.4).
itexists,EVICPRESS’sretrieverfetchestheKVcachefrom
4.1 Overallworkflow
|     |     |     |     |     | the location |     | that the lookup | module | returns. If it does not, |
| --- | --- | --- | --- | --- | ------------ | --- | --------------- | ------ | ------------------------ |
EVICPRESScontainstwophases:aninitialofflinephaseand
theKVcacheisgeneratedbytheLLM,andthen savedby
anonlineinferencephase. EVICPRESS to the remote disk,which is assumed to have
Initialofflinephase: Duringtheinitialofflinephase,EVIC- unlimitedspace.Astheonlinequeriesmaydifferfromthe
PRESS’sprofiler(§4.3)computestheutilityfunctionscores initialsetofprofilequeries,EVICPRESS’sprofilingmodule
(§4.2)forallpossibleeviction-compressionconfigurationsof
alsoperformsperiodicre-profilingbasedonanupdatedsetof
eachcontext.Tocomputethequalityanddelayitemsinthe queriesforeachcontext,whentheachievedqualityislower
utilityfunction,weuseaninitialsetofgeneratedquestions thanthresholdX%andfreeGPUcyclesareavailable.
foreachcontext.
NotethattheKVcachestoringoperationrunsontheCPU,
Onlineinferencephase: Intheonlineinferencephase,when whichdoesnotblocktheonlineinferencecomputationsfor
auserrequestcomesin,thelookupmodulefirstdetermines other queries. Furthermore, the re-profiling step first com-
whether the KV cache for the requested context exists in pressestheKVcacheintomultipleversionsandrunsdecod-
the multi-tier storage system managed by EVICPRESS. If ingphaseswithdifferentKVcacheversions.Boththecom-
5

|     |         | KVcache   |     | worsethan(keydiff,0.9,CPU)astheformeronehasthesame   |     |     |     |
| --- | ------- | --------- | --- | ---------------------------------------------------- | --- | --- | --- |
|     | KVcache | KVcache   |     |                                                      |     |     |     |
|     |         |           | CPU | TTFTandlowerquality.Thisisreflectedbytheutilityfunc- |     |     |     |
|     | exists  | retriever |     |                                                      |     |     |     |
LocalSSD
Lookup SavinggeneratedKV tionscoretoo,withtheformeronehavingalowerscorethan
module
| User |     | cachetoremotedisk | RemoteDisk | thelatter. |     |     |     |
| ---- | --- | ----------------- | ---------- | ---------- | --- | --- | --- |
KVcache
Thesystemenablesuserstonavigatethetrade-offbetween
| InferencePhase | doesn’texist |     | Updated |     |     |     |     |
| -------------- | ------------ | --- | ------- | --- | --- | --- | --- |
Non- PeriodicallyfetchingallKVcache eviction- generation quality and latency with the tunable parameter
blockingto andnewqueries compression α.Alargerαprioritizesgenerationquality,whileasmaller
configurations
inference
αprioritizessmallerTTFT.Forexample,inthe2wikimQA
| Profiler |     | Choosingeviction- |     |     |     |     |     |
| -------- | --- | ----------------- | --- | --- | --- | --- | --- |
compression dataset,acompressionratioof0.4yieldsaqualityscoreof
| module(4.3) | Utilityfunctionscores |     |     |     |     |     |     |
| ----------- | --------------------- | --- | --- | --- | --- | --- | --- |
configuration(4.4) 0.82 and a TTFT of 0.20 seconds, whereas a ratio of 0.2
(4.2)
improvesqualityto0.92butincreasesTTFTto0.24seconds.
Consequently,ifausersetsα>0.3,theutilityfunctionfavors
| Figure6:End-to-endworkflowof |     |     | EVICPRESS. |                    |          |                   |               |
| ---------------------------- | --- | --- | ---------- | ------------------ | -------- | ----------------- | ------------- |
|                              |     |     |            | the higher quality | provided | by the 0.2 ratio; | conversely, a |
lowerαprioritizesthelatencyreductionofferedbythe0.4
| pressionstepanddecodingcomputationsarelightweightin |     |     |     | ratio. |     |     |     |
| --------------------------------------------------- | --- | --- | --- | ------ | --- | --- | --- |
termsofGPUcomputations,andcanbebatchedwithotherre-
Anotherimportantdesignquestionwhencalculatingthe
quests’decodingcomputations.Aswewillshowsoon(§6.3), utilityfunctionis:whatisasuitablegranularitytocompute
re-profilingincursminimalsystemoverhead.
|     |     |     |     | this utility function | on? EVICPRESS | computes | this utility |
| --- | --- | --- | --- | --------------------- | ------------- | -------- | ------------ |
functionbasedoncontext-level,insteadofchunk-levelwhich
| 4.2 Utility | function: | Balancing | quality and |                  |           |               |                  |
| ----------- | --------- | --------- | ----------- | ---------------- | --------- | ------------- | ---------------- |
|             |           |           |             | splits the whole | context’s | KV cache into | multiple chunks, |
loadingdelay
ortoken-level.Thisisbecausecontext-levelprofilinghasa
Eachcompressionandevictionconfigurationpresentsapoint globalviewofimportanttokendistributionconsideringtheen-
on the generation quality and loading delay trade-off. To tirecontext.Ontheotherhand,forchunk-levelprofiling,some
easilymeasuretheimpactofeachconfigurationonquality tokensmaybeimportantwithinthescopeofachunk,while
andloadingdelay,wedesignautilityfunctionwhichoutputs unimportantatthescopeofthewholecontext.Token-level
asinglescoreforeacheviction-compressionconfiguration– profilingisevenmoresuboptimalsincequalitymeasurement
thestoragetierthattheevictionschemeoffloadsKVcache isnotfeasible.Andtoken-levelretrievalcannotsaturateband-
to,andthecompressionmethodandratioitapplies: widthandleadstolowspeedandhighoverheadbecauseof
inconsistentmemoryaccess.
Util(method,ratio,device)=(α·quality−TTFT)·frequency,
whereαisthetrade-offparameterbetweenqualityandload- 4.3 Profiler module: Computing the utility
ingdelay—higherαimprovesthequalityofEVICPRESSat function scores and choosing eviction-
thecostofhigherloadingdelayinaverage,andinoureval-
compressionconfiguration
uationwewillvaryαtoshowthetrade-offofEVICPRESS
betweenqualityandloadingdelay. Basedoninsight1,thecompression-evictionconfiguration
Intheaboveequation,theinputtothefunctionisthestor- mustadapttodifferentcontexts.Insight2indicatesthatthe
age device tierwhichthe eviction algorithm sends the KV configurationcannotsimplybedeterminedbythecharacter-
cacheto,andthecompressionmethod/ratiothatisapplied. isticsofthecontext,suchaslengthsortypes.Thus,instead
Thequalityiscomputedbasedonthecompressionmethod of coming up with context characteristics that indicate the
andratio,averagedacrossallthedataintheprofilingquery benefitsofeachconfiguration,EVICPRESSdirectlyprofiles
set; the delay item is computed with estimated latency to theimpactofallconfigurationsonoverallqualityanddelay.
fetcha(compressed)KVcachefromthecorrespondingstor- Specifically,EVICPRESSusestheutilityfunction(§4.2)that
agedevice;andthefrequencyitemisthenumberoftimesthe outputsasinglescoreforeachconfigurationof(storagede-
vice,compressionmethod,compressionratio)tocaptureboth
contextisaccessed.
| To make | it concrete, | take the examples | from §3.1, | delayandquality. |     |     |     |
| ------- | ------------ | ----------------- | ---------- | ---------------- | --- | --- | --- |
whenusingkeydiffcompression,acrossthecontextsfrom Beforetheonlineinferencestage,asthesystemdoesnot
2wikimQA dataset which has highersensitivity,with com- seeanyqueriesbefore,theprofilingmodulegeneratesasetof
pression ratio of 0.6,we can reach a quality score of 0.65 trainingquestionsforeachcontextfollowingpriorworks[10,
andTTFTof0.05seconds;whileacrosscontextsfromNarra- 47–49,57],tocomputetheaveragequalityanddelayonthem.
tiveQAdatasetthathaslowersensitivity,withcompression ThequestionsaregeneratedbytheGPT-5APIaboutdifferent
ratio of 0.9,we can get a quality score of 0.67 with TTFT paragraphs in the contexts, aiming to obtain the eviction-
of 0.05 seconds. Assuming α=1 and equal frequency for compressionconfigurationthatdropsasmuchredundancyfor
eachcontext,theutilityfunctionscoreis0.607and0.621re- eachparagraphaspossibletoreducedelay.
spectively.Theconfigurationof(keydiff,0.6,CPU)isstrictly Duringonlineinference,as real-worldqueriesmaydrift
6

fromthetrainingqueries,theprofilingmodulealsoperiod- numberofhittokensisreturned,andthevLLMengineallo-
ically re-profiles with a newer dataset. The re-profiling is catestheamountofGPUmemoryforstoringtheKVcachefor
retrieve(tokens)->KV
triggeredwheneverthe quality achievedon the online test- the hit tokens. Then,LMCache’s
ing dataset is below the quality achieved on the profiling CachefunctionistriggeredtoloadtheKVcachefromEVIC-
datasetbyathresholdofX%,andwhenthesystemhasfree PRESS-managedmulti-tierstoragebackend,andputitinto
GPU cycles. Note that the profiling process only requires thepagedGPUmemoryallocatedbyvLLM.
theKVcachetobecompressedwithdifferentmethodsand If it does not find the requested KV cache, LM-
ratios,whichincursveryminimaloverhead,andrunningthe Cache’s store(tokens, KV Cache) function is triggered
decodingphasewiththesedifferentcompressedversions.The to save the generated KV cache to CPU DRAM, and
| decodingphaseisknowntobelargelybatchable(batching |     |     |     |     |     |                |     |           | manage |           |       |        |
| ------------------------------------------------- | --- | --- | --- | --- | --- | -------------- | --- | --------- | ------ | --------- | ----- | ------ |
|                                                   |     |     |     |     |     | then EVICPRESS |     | calls the |        | function, | which | places |
withotherrequestsdoesnotincreasedecodingdelay).Thus, the new KV cache on a storage device and updates ex-
availableGPUcyclesoccurwhenthenumberofonlineinfer- isting KV cache entries if the device is full. Specifi-
encerequestsissmallerthanthemaximumbatchsizeinthe cally,we use a class KVConfig(KV Cache) which records
system.Inpractice,userscanfreelyadaptX%basedontheir the utility function scores for all possible configurations
| qualityrequirements. |     |     |     |     |     |                 |          |         |     | manage           |             |       |
| -------------------- | --- | --- | --- | --- | --- | --------------- | -------- | ------- | --- | ---------------- | ----------- | ----- |
|                      |     |     |     |     |     | for the         | given KV | cache.  | The |                  | function    | takes |
|                      |     |     |     |     |     | in [KVConfig(KV |          | Cache   | 1), | ..., KVConfig(KV |             | Cache |
|                      |     |     |     |     |     | n)] and         | computes | (device |     | placement,       | compression |       |
4.4 Eviction-compressionconfigurationselec-
|     |     |     |     |     |     | method, | compression |     | ratio) | for each | KV  | cache that |
| --- | --- | --- | --- | --- | --- | ------- | ----------- | --- | ------ | -------- | --- | ---------- |
tionmodule
|     |     |     |     |     |     | needs updating,calculated |     |     | by EVICPRESS’s |     | configuration |     |
| --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | -------------- | --- | ------------- | --- |
selectionmodule(§4.4).Finally,themanagefunctionupdates
Givenaninitialsetofcontexts,whereeachcontexthasaset
thoseKVcacheswiththenewconfiguration.
ofgeneratedqueriesforprofiling,EVICPRESScomputesthe
utilityfunctionscoresforallpossibleeviction-compression
configurationsoneachcontext.EVICPRESS’sconfiguration
6 Evaluation
selectionmodulethenplacesthe(compressed)KVcacheto
acertainstoragetierforeachcontextwiththeconfiguration
Thekeytakeawaysfromtheevaluationare:
correspondingtothehighestutilityfunctionscore.
When a storage device S is full, existing KV cache on • Across5modelsand555contextsfrom12datasetsfrom
thatdeviceneedstobeupdated.Specifically,eachcontext’s LongBench [3], EVICPRESS reduces TTFT by 1.43 to
KV cache can be updated either by more aggressive com- 3.77×andimprovesqualityby13.58%to55.40%atthe
| pression to | fitin S,orby | evicting | itto | a lower-tierstorage |     |     |     |     |     |     |     |     |
| ----------- | ------------ | -------- | ---- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
sameTTFT,comparedtothebaselinethatcombinesfixed
to save space. Each context has (#compressionmethods× compressionandclassiceviction.
| #compressionratios | ×   | #lower-tierstoragedevices) |     |     | of op- |     |     |     |     |     |     |     |
| ------------------ | --- | -------------------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
• ComparedtosmartKVcacheevictionsystemIMPRESS
tionstochoosefrom.Here,wefilteroutoptionsthatkeepKV [5], EVICPRESS reduces TTFT by 1.5 to 5.2 × and im-
cacheonstorageSbutdonotsavespace(suchaskeepingthe provesqualityby14.29%to27.00%atthesameTTFT.
compressionratiosandnotevictingtothenexttier).
|     |     |     |     |     |     | • Compared | to full | prefill | or full | eviction, | EVICPRESS | re- |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | ------- | ------- | --------- | --------- | --- |
Theproblemofdecidingtheconfigurationforeachcontext
ducesTTFTby1.29to2.19×within3%ofqualitydrop.
whilemaximizingthetotalutilityfunctionscoreondevice
| S is a Multi-Choice | Knapsack |     | Problem,which |     | is NP-hard. |     |     |     |     |     |     |     |
| ------------------- | -------- | --- | ------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Wegreedilysolvethisproblembyrepeatingthegreedypro- 6.1 Evaluationsetup
cessoffindingupdatedconfigurationswiththelowestutility
scoredropforcontextsonS,untiltheupdatedKVcachefits
|     |     |     |     |     |     | Dataset |     | Mean | Std Topic |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --- | ---- | --------- | --- | --- | --- |
successfully.Then,thesameupdateprocessisappliedrecur-
|     |     |     |     |     |     | narrativeqa |     | 108K | 55K Longstories(literature,film). |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | ---- | --------------------------------- | --- | --- | --- |
sivelyifthelower-tierstorageisalsofull.Theexactgreedy qasper 24K 12K NLPresearchpapers
algorithmforthisprocessisdescribedin§A. multifieldqa_en 29K 15K Legaldocumentsandreports.
|     |     |     |     |     |     | hotpotqa |     | 57K | 18K Wikipediaarticles. |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | --- | --- | ---------------------- | --- | --- | --- |
|     |     |     |     |     |     | 2wikimqa |     | 30K | 15K Wikipediaarticles. |     |     |     |
|     |     |     |     |     |     | musique  |     | 69K | 9K Wikipediaarticles.  |     |     |     |
5 Implementation
|     |     |     |     |     |     | gov_report |     | 54K | 34K USGAOandCRSreports.       |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | --- | ----------------------------- | --- | --- | --- |
|     |     |     |     |     |     | qmsum      |     | 57K | 27K Meetingscriptsandrecords. |     |     |     |
We implement EVICPRESS on top of vLLM v0.11.2 and multi_news 12K 10K Newsarticlesonthesameevent.
|         |              |      |       |          |            | trec     |     | 30K | 12K Newsarticlesandwebpages. |     |     |     |
| ------- | ------------ | ---- | ----- | -------- | ---------- | -------- | --- | --- | ---------------------------- | --- | --- | --- |
| LMCache | v0.3.9post2, | with | about | 3K lines | of code in |          |     |     |                              |     |     |     |
|         |              |      |       |          |            | triviaqa |     | 47K | 25K Triviaquestions.         |     |     |     |
PythonbasedonPyTorchv2.9.
|                                            |         |              |                 |     |       | samsum |     | 34K | 17K Messenger-likeconversationhisto- |     |     |     |
| ------------------------------------------ | ------- | ------------ | --------------- | --- | ----- | ------ | --- | --- | ------------------------------------ | --- | --- | --- |
| Whenanewtextquerycomesin,wecallLMCache’sKV |         |              |                 |     |       |        |     |     | ries.                                |     |     |     |
| cache lookup                               | module, | specifically | lookup(tokens). |     | If it |        |     |     |                                      |     |     |     |
findsthattherequestedKVcacheexistsin thesystem,the Table2:DatasetStatisticsanddescription.
7

B
etter B etter B etter
B B
etter etter
Figure7: EVICPRESS reducesTTFTby1.43to3.77×withthesamequalityscorecomparedtocompression+LRU-based
eviction,andreducesTTFTby1.22to1.56×comparedtoprefillandevictiononlyscheme.
LLMs: We cover a wide range of popular open-source recentlyused(LRU)policy,withoutanycompression.
LLMs, including dense LLMs, including Llama (meta- • (keydiff/knorm/snapkv) Compression + LRU-based evic-
llama/Llama-3.1-8B-Instruct),Qwen(Qwen/Qwen2.5-14B- tion:Thisbaselineapplieseitherkeydiff[35],knorm[9]
Instruct),LongChat(lmsys/longchat-7b-v1.5-32k)andMis- orsnapkv[27]compressiononallcontexts.Wealsovary
tral(mistralai/Mistral-7B-Instruct-v0.3),andalsoMixture-of- the compression ratio foreach method,whichforms the
ExpertLLM(Qwen/Qwen3-30B-A3B-Instruct-2507). quality-TTFT trade-off. At a certain compression ratio,
Dataset: We use LongBench benchmark [3],which com- whenastoragetierisfull,thesystemevictsthecompressed
binesmanylong-contextdatasetsandincludesawiderange KV cache to lower-tierstorage withLRU. Among these
ofcontextgenres(includinggovernmentreports,news,con- methods,keydiffcomputestheaveragecosinesimilarityof
versationsandstories),asourevaluationdataset.Thedataset everytokentoallothertokens,anddropsthetokenswhose
statistics,includingthemeanandstandarddeviationofthe similarities are high among all; knorm computes the L2
contextlengths,andthedatasetdescriptionislistedinTable2. normofthekeycache,anddropsthosewithlowL2norm;
snapkvcomputesthecrossattentionbetweentokensnear
Queryconstruction: Werandomlysampled555contexts
thequeryandtheearliercontexts,anddropsthosetokens
fromlongbenchdataset.Foreachcontext,wepromptsgpt-5
withlowcross-attentionscores.
togenerate100QAqueries(followingpreviousworks[10,
• IMPRESS:IMPRESS[5]usesthreeoutofeightattention
47–49,57]), 50 as training queries to build the profiler of
headstocomputetheattentionscorestoprofileimportant
EVICPRESSand50fortesting.
tokensintheKVcache.IMPRESSkeepsX%ofimportant
Hardware: We run our experiments on one 80GB H100
tokenswhoseattentionscoresareamongthetopX%ofall
GPUthathoststhemodelweightsforeachmodel.Weallocate
tokens.ThenitsplitseachKVcacheintochunks,andonly
80GBCPUDRAM,800GBSSD,andassumethattheremote
loadsKVcachechunkscontainingimportanttokensduring
storagehasunlimitedspace.
inference.WevaryX%inourexperimentstoachievethe
Baselines: WecompareEVICPRESSwithbaselinesthatapply quality-delaytrade-offofIMPRESS.
KVcachecompressionoreviction.
Evaluationmetrics:
• Prefill: This baseline performs full prefill on each query • Qualityscore:measuresthesimilarityofanswersbasedon
withoutprefixcaching.WerunthiswithvLLMv0.11.2. compressedKVcachevs.answersbasedonuncompressed
• Evictiononly:Whenastoragetierbecomesfull,thissystem KVcache.Followingpriorwork[2,12,40],wecomputethe
evicts KV cache to the next storage tier using an least- cosinesimilaritybetweentheembeddingofthegenerated
8

keydiff+LRU-based eviction knorm+LRU-based eviction snapkv+LRU-based eviction
|     |     | impress |     |     |     | eviction only |     | prefill |     |     |     |
| --- | --- | ------- | --- | --- | --- | ------------- | --- | ------- | --- | --- | --- |
ours
Llama-3.1-8B-Instruct Mistral-7B-Instruct-v0.3 Qwen2.5-14B-Instruct
)s( TFTT egarevA
|     | 4    |     |      |     |      | 4   |          | 4    |         |     |     |
| --- | ---- | --- | ---- | --- | ---- | --- | -------- | ---- | ------- | --- | --- |
|     | 2    |     |      |     |      | 2   |          | 2    |         |     |     |
|     | 0    |     |      |     |      | 0   |          | 0    |         |     |     |
|     |      | 0   | 5 10 | 15  |      | 0   | 10 20 30 | 0.0  | 2.5 5.0 | 7.5 |     |
|     | 0.20 |     |      |     | 0.20 |     |          | 0.20 |         |     |     |
)s( LTI egarevA
|     | 0.15 |     |      |     | 0.15 |     |          | 0.15 |         |     |     |
| --- | ---- | --- | ---- | --- | ---- | --- | -------- | ---- | ------- | --- | --- |
|     | 0.10 |     |      |     | 0.10 |     |          | 0.10 |         |     |     |
|     | 0.05 |     |      |     | 0.05 |     |          | 0.05 |         |     |     |
|     | 0.00 |     |      |     | 0.00 |     |          | 0.00 |         |     |     |
|     |      | 0   | 5 10 | 15  |      | 0   | 10 20 30 | 0.0  | 2.5 5.0 | 7.5 |     |
|     |      |     | QPS  |     |      |     | QPS      |      | QPS     |     |     |
Figure8:EVICPRESSmaintainsconsistentlylowTTFTandITLacrossallQPSlevels.
6.2 Evaluationresults
| compression+LRU-based eviction |     |     |     | impress |     | ours |                            |     |                      |     |     |
| ------------------------------ | --- | --- | --- | ------- | --- | ---- | -------------------------- | --- | -------------------- | --- | --- |
|                                |     |     |     |         |     |      | TTFTv.s.generationquality: |     | InFigure7,comparedto |     |     |
erocs ytilauq egarevA
| 0.72 |     |     | 0.80 |     |     |     | thebaselinesthatapplieskeydiff,knormandsnapkvand |     |     |     |     |
| ---- | --- | --- | ---- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- |
| 0.70 |     |     | 0.74 |     |     |     | LRU-basedeviction,EVICPRESSreducesTTFTby1.43to   |     |     |     |     |
3.77×atthesamequalityscore,andimprovesqualityscores
0.68
|     |     |     | 0.68 |     |     |     | by13.58%to55.40%atthesameTTFT.Comparedtofull |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- |
0.66 0.62 prefill that does not affect generation quality, EVICPRESS
0 10 20 30 40 0 20 40 60 80 reducesTTFTby1.29to2.19×within3%ofqualitydrop.
|     | CPU-hit request % |     |     | SSD-hit request % |     |     |     |     |     |     |     |
| --- | ----------------- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
ComparedtoLRU-basedeviction,EVICPRESSreducesTTFT
|     |     |     |     |     |     |     | by 1.22 to | 1.56× with quality | drop less than | 3%. | Finally, |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------------------ | -------------- | --- | -------- |
Figure9:WithQwen2.5-14B-Instruct,EVICPRESShashigher comparedtoIMPRESS,EVICPRESS reducesTTFTby1.5
fastdevice(CPU)hitrequestpercentagewhilemaintaining to 5.2× withoutdegrading generation quality,orimproves
higherquality. generationqualityby14.29%to27.00%withoutincreasing
TTFT.
|     |     |     |     |     |     |     | TTFTandITLv.s.QPS: | Figure8showstheTTFTandITL |          |           |       |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | ------------------------- | -------- | --------- | ----- |
|     |     |     |     |     |     |     | under different    | query-per-second          | (QPS),at | a quality | score |
answerandthatoftheanswerbasedonuncompressedKV
targetof80%forLlama-3.1-8B-Instruct,Mistral-7B-Instruct-
cacheusingtheMiniLM-L6-v2model.Thehigherthisscore
|     |     |     |     |     |     |     | v0.3 and Qwen2.5-14B-Instruct. |     | At the same | TTFT | level, |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | ----------- | ---- | ------ |
is,thecloserthegeneratedanswersiscomparedtoanswers
EVICPRESSachieves2.0to3.6×higherrequestprocessing
onuncompressedKVcache,whichisthebetter.
rate,comparedtothebestbaseline.Similarly,atthesameITL
• Time-to-first-token(TTFT)isthetimebetweenthearrival
level,EVICPRESSachieves1.9to3.4×higherQPScompared
| of a query | to  | the time | of the first | generated | token. | This |     |     |     |     |     |
| ---------- | --- | -------- | ------------ | --------- | ------ | ---- | --- | --- | --- | --- | --- |
tothebestbaseline.
includesthedelaytoretrieveKVcachefromthestorage
|     |     |     |     |     |     |     | EVICPRESS’simprovement: |     | EVICPRESSoutperformsdif- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | ------------------------ | --- | --- |
devices,andthetimetoprefillthenewqueries.
ferentbaselinesfordifferentreasons.First,comparedtothe
• Inter-token-latency(ITL)istheaveragedelayforgenerating baselinesthatapplysamecompressionmethodsandratioson
consecutivetokens.ITLreflectstheencodingthroughput,
|     |     |     |     |     |     |     | allcontexts,EVICPRESS | isbetterasthecontextsensitivity |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | ------------------------------- | --- | --- | --- |
andincreaseswhentheinferenceenginehasmorerunning differenceisconsideredandeviction-compressionconfigu-
requests.
rationisadjustedfordifferentcontexts.Secondly,compared
• End-to-endlatencyisthetimebetweenwhentherequest to LRU-based eviction, which evicts all contexts that can-
comesinandwhenthelasttokenisreceived. notfitontheCPUDRAMtolowertierstorage,EVICPRESS
9

0.13
0.09
0.05
erocs
ytilauQ
4
2
0
2.5 3.0 3.5 4.0 4.5
Minutes since start
)s(
ycnetaL
6.3 Microbenchmarks
Difference (with without periodic update)
The effect of periodic update: Figure 10 evaluates our
periodic-profilingmechanism(§4.3)usingrealtimestamps
from the Azure inference trace [46]. As Azure inference
trace does not include the concrete content of the request,
weusetherequestwegenerated(describedin§4.3)instead.
EVICPRESS then performs profiling when the discrepancy
betweentheprofiledqualityandthepredictedqualityofthe
querydecodedwiththestoredKVcacheexceeds0.3.
We illustrate the time series of using periodic profiling
versusnotinFigure10:
• Thefigureaboveshowshowthequalitygainofusingperi-
odicprofilingversusnotevolvesovertime.Fromthisfigure,
Figure10:QualityandlatencyovertimeunderAzurework- wecanseethatthequalitygainincreasesgraduallyover
load[41,46,58]withonlyonetrainingquerypercontext.Red timeandeventuallysaturatesataround11%,becauseperi-
verticallinesdenoteperiodicprofilingeventswithLlama-3.1- odicprofilingallowsEVICPRESStouseup-to-dateprofile
8B-Instruct.Frequentprofilingatthebeginningandaround tomakenear-optimalcompressiondecision,whilethepro-
3.75–4 minutes introduces brieflatencyspikes,butlatency fileofnotdoingperiodicupdategraduallygetsoutdatedand
quicklyreturnstobaseline.Asprofilingcorrectsstalecom- resultinloweraccuracyunderthesamestorageresource
pressiondecisions,modelqualitysteadilyimprovesthrough- budget.
outexecution.
• The figure below shows how the end-to-end latency of
requestsofEVICPRESSvaariesovertime.Wecanseethat,
EVICPRESShashigherend-to-endlatencyintheperiodic
profilingintervals(theredintervals)in average,because
isbettersincesomelow-sensitivitycontextscanbelargely periodicprofilingrequiresGPUresourcesandslowdown
compressedwithoutbeingevictedtoslowstorage.Thirdly, the execution of existing requests. That said,the quality
comparedtofullprefillwhichincurssignificantamountof improvement brought by periodic profiling (about 10%)
GPUcomputation,EVICPRESSisbettersincetheprefillcom- stilljustifiesitscost.
putationisskipped.Finally,comparedwithIMPRESS,which Different context needs different eviction-compression
splits KV cache into chunks,andnotonly loads important configurations: Figure11visualizesthedistributionofse-
tokens,butalsounimportanttokensineachchunktofastde- lectedeviction-compressionconfigurations.Weobservethat
vice,EVICPRESSonlyputsKVcachethatmaximizethetotal thesystemindeedmakesuseofadiversesetofcompression
utilityfunctionscoresonfastdevice. configurations,indicatingthatalargesearchspaceisnotonly
necessarybutactivelyexploited.Wehighlighttwoinsights:
Insummary,EVICPRESSachievesmuchhighercachehit
• KVcachesondiskactuallyrequireshighercompression
rateinCPUDRAM,asshowninFigure9,comparedtobase-
ratio in average compared to KV caches on CPU. This
lines that apply the same compression methods and LRU-
is because KV caches on disk are slowerto read,which
basedevictiononallcontextsandIMPRESS.Ontheother
requiresEVICPRESStofurthercompresstheKVcachesto
hand,asshownintherighthandsideofthefigure,thecache
reducetheirloadingtime.
hitrateonlocalSSDforEVICPRESSislowerthanotherbase-
• Wenotethat,thoughkeydiff generallyoutperformssnapkv
lines,whichillustratesthatmostofthecachehittokensreside
(whichin turn outperforms knorm),yet a non-negligible
onCPUDRAM,providingmuchfasterloadingdelaythan
portion ofKV caches stilladopts snapkv due to context-
baselines.
specificbenefits.
WeacknowledgethattheimprovementofEVICPRESSde- Improvementsacrossheterogeneouscontexts: Figure12il-
pendsonthesizesofKVcachefordifferentLLMs.Whenthe lustratestheTTFT—Averagescoretrade-offofEVICPRESS
KVcacheissmall,DRAMcanholdmosttokens,leavinglittle andbaselinesacrossdifferentsubtasksoflongbenchdataset.
roomforimprovement.Forexample,theMoEmodelQwen3- Wedrawtwoconclusions:
30B-A3B-Instruct-2507hasasmallKVcache(0.0915 GB • EVICPRESS hashigherimprovementontasksthatcorre-
per1Ktokens),whileasmallerdensemodelsuchasLlama- spondstoshortercontexts(e.g.,multi_newsandqasper).
3.1-8B-Instructhas0.12GBKVcacheper1Ktokens,soeven Thisisbecause,inordertoachievehigherCPUcachehit
withaconservativecompressionratio,mostoftheKVcache rate,EVICPRESSputsshortercontextsinCPUinsteadof
canbestoredinCPUDRAM,leadingtolowTTFT. longerones,sothattheCPUcanfitintotheKVcachesof
10

termineimportanttokensfromtheKVcache,andevictsnon-
|     |     |     | CPU |     |     |     | SSD |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
importantonestolower-tierstorage[5,50].However,thisline
 rednu ehcac VK # ofworkonlyappliesdifferentevictionschemesonKVcache,
etar .pmoc hcae 20
|     |     |     |     |     | 40  |     |     |     | notevenconsideringcombiningwithcompressionschemes |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
toachievehighercachehitrates.
|     |     |     |     |     |     |     |     |     | KVcachecompression: |     |     | Toachievehighercachehitrate, |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ---------------------------- | --- | --- | --- |
|     | 10  |     |     |     | 20  |     |     |     |                     |     |     |                              |     |     |     |
manyworkshaveexploredorthogonalmethodstocompress
|     |     |         |         |     |     |     |         |     | KV caches   | to reduce    | memory    | footprint.       |      | Token dropping |        |
| --- | --- | ------- | ------- | --- | --- | --- | ------- | --- | ----------- | ------------ | --------- | ---------------- | ---- | -------------- | ------ |
|     | 0   |         |         |     | 0   |     |         |     | methods     | drop less    | important | tokens           | from | the KV         | cache, |
|     |     | 0.0 0.1 | 0.2 0.3 | 0.4 | 0.0 | 0.2 | 0.4 0.6 | 0.8 |             |              |           |                  |      |                |        |
|     |     |         |         |     |     |     |         |     | eitherbased | on attention |           | scores orlearned |      | importance     | sig-   |
200
nals[6,38,59].QuantizationmethodsreducethesizeofKV
 rednu ehcac VK # dohtem .pmoc hcae
|     | 20  |     |     |     | 150 |     |     |     | cachebyusingfewerbitstorepresenteachKVcacheelement, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
includingadaptivequantizationwheredifferentlayersareap-
|     |     |     |     |     | 100 |     |     |     | pliedwithdifferentlevelsofquantization[17,31]anduniform |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
10
|     |     |     |     |     |     |     |     |     | quantization | [32]. | Similarly,this | line | of work | only | consid- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | -------------- | ---- | ------- | ---- | ------- |
50
ersthepossibilityofcompression,withoutcombiningitwith
0 0 eviction,even when the quality drop is high. Another line
m p. diff pkv or m diff pkv m p. o f w o r k a p p l i e s d i ffe r e n t l ev e l s o f c o m p re s sio n c o n s i d e r in g
|     | o   | ey  | a n |     | ey  | a   | o   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
o c k s n k k s n o c th e e rr o r s e n s i t iv it y o f e a c h c o n t ex t [ 7 ,5 1 ]. H ow e v e r , i t d o e s
|     | n   |     |     |     |     |     | n   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
notconsiderper-contextsensitivitydifferencetotheproblem
|     |     |     |     |     |     |     |     |     | ofjointoptimization |     | between | eviction | andcompression |     | as  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ------- | -------- | -------------- | --- | --- |
EVICPRESSdoes.
Figure11:Distributionofselectedcompressionmethodsand
compressionratesacrossCPUandSSDstoragetiersunder FasterLLMserving: Anothercomplementarylineofwork
α=1 with Mistral-7B-Instruct-v0.3. EVICPRESS actively speedsupLLMserving bybetterscheduling toreduce the
usesabroadcombinationofconfigurations,validatingthe bubblesacrossdifferentinferencestages(e.g.,prefill,decode
needforalargesearchspace.
ortool-calling)[1,8];prefill-decodedisaggregationtoplace
|     |     |     |     |     |     |     |     |     | the prefill | and decode | phases | to different |     | GPUs to | reduce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | ------ | ------------ | --- | ------- | ------ |
SLOviolationrate[36,61];speculativedecodingapproaches
morerequests.
whichproposesusingasmaller,fasterdraftmodeltopropose
| •   | EVICPRESS | improves |     | the performance |     | for | 11 out | of 12 |     |     |     |     |     |     |     |
| --- | --------- | -------- | --- | --------------- | --- | --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- |
asequenceoffuturetokens,andusingthelargertargetmodel
|     | tasks. | The only | exception | is  | on musique, |     | where | EVIC- |     |     |     |     |     |     |     |
| --- | ------ | -------- | --------- | --- | ----------- | --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
toverifythetokensinparallel[25,33];andchunked-prefillto
PRESSperformsslightlyworsethanthekeydiffwithLRU
enablepiggy-backeddecodingandreducememoryconsump-
evictionbaseline.
|     |     |     |     |     |     |     |     |     | tion ofprefill | [1]. | This | line ofworkis | complementaryand |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ---- | ---- | ------------- | ---------------- | --- | --- |
7 RelatedWork orthogonaltoEVICPRESS,andcanbecombinedwithEVIC-
PRESStofurtherincreasetheefficiencyofLLMinference.
| KVcachereuseandevictionsystems: |     |     |     |     |     | Storingandreusing |     |     |                                   |     |     |     |     |                |     |
| ------------------------------- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | -------------- | --- |
|                                 |     |     |     |     |     |                   |     |     | OtherKVcacherelatedoptimizations: |     |     |     |     | Therearealotof |     |
KV cache across multiple requests has been a critical fo- otherworksthatoptimizeKVcacheinLLMinference,includ-
| cus | in the | design | of high-performance |     |     | LLM | serving | sys- |     |     |     |     |     |     |     |
| --- | ------ | ------ | ------------------- | --- | --- | --- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
ingnon-prefixKVcachesharingwithapproximation[14,54],
tems. This includes paged memory management in GPU, cross-LLM KV cache reusing with partial layer recompu-
suchasvLLM[24]andSGLang[60].However,thislineof tation [30], request routing that is aware of KV cache lo-
| work | suffers | from | low cache | hit | rate | when | the request | ar- |              |           |     |                 |     |                |     |
| ---- | ------- | ---- | --------- | --- | ---- | ---- | ----------- | --- | ------------ | --------- | --- | --------------- | --- | -------------- | --- |
|      |         |      |           |     |      |      |             |     | cation [43], | improving |     | paged attention | for | multi-modality |     |
rivalrateishighandthenumberofconcurrentusersislarge, modelswithdifferentshapesofKVcache[44],andmanaging
| since | GPU | memory | only | has limited |     | space | (40 to | 80 GB |     |     |     |     |     |     |     |
| ----- | --- | ------ | ---- | ----------- | --- | ----- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- |
KVcacheforsparse-attentionmodels[4].Whilethislineof
forA100orH100GPUs).Thus,moreworkisproposedto workalsomakesKVcacheinLLMinferencemorepractical
evictKVcachethatcannotfitinGPUmemorytolower-tier inmultipleusecases,EVICPRESSiscomplementarytothem
storagedevices,suchasCPUDRAM[26,29,52,60],local
astheyjointlyoptimizeevictionandcompression.
disks[22,29,56],andevenremotedisks[22,29].Forexample,
LMCache[21,29]reusessplitseachKVcacheintomultiple
|     |     |     |     |     |     |     |     |     | 8 Limitationsandfuturework |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- |
chunks,andoptimizestheloadingspeedfromCPU/disksto
GPUmemorywithhigh-performanceCUDAkernels;Moon- Deploymentscopeandassumptions: EVICPRESStargetsa
cake[37]andInstInfer[34]optimizestheloadingbandwidths commonandchallengingdeploymentscenariowithlimited
across different GPU clusters and performance of prefill- GPU and CPU memory and a multi-tier storage hierarchy,
decodedisaggregation;IMPRESSappliesfine-grainedtoken- whereKVcachesroutinelycontendforcapacity,andeviction
leveleviction,whichusesasubsetofattentionheadstode- orcompressiondecisionsconcretelyaffectend-to-endlatency
11

keydiff+LRU-based eviction knorm+LRU-based eviction snapkv+LRU-based eviction
|     | prefill  |     |     | ours       |     |          |     |     |            |     |
| --- | -------- | --- | --- | ---------- | --- | -------- | --- | --- | ---------- | --- |
|     | 2wikimqa |     |     | gov_report |     | hotpotqa |     |     | multi_news |     |
| 1.0 |          |     | 1.0 |            | 1.0 |          |     | 1.0 |            |     |
erocs egarevA
| 0.8 |                 |     | 0.8 |         | 0.8 |             |         | 0.8 |        |     |
| --- | --------------- | --- | --- | ------- | --- | ----------- | ------- | --- | ------ | --- |
| 0.6 |                 |     | 0.6 |         | 0.6 |             |         | 0.6 |        |     |
| 0.4 |                 |     | 0.4 |         | 0.4 |             |         | 0.4 |        |     |
| 0.2 |                 |     | 0.2 |         | 0.2 |             |         | 0.2 |        |     |
| 0.0 |                 |     | 0.0 |         | 0.0 |             |         | 0.0 |        |     |
|     | 0.0 0.1         | 0.2 | 0.0 | 0.2     | 0.0 |             | 0.2 0.4 | 0.0 | 0.1    |     |
|     | multifieldqa_en |     |     | musique |     | narrativeqa |         |     | qasper |     |
1.0
| erocs egarevA 1.0 |         |     | 1.0 |         |     |     |      | 1.0 |          |     |
| ----------------- | ------- | --- | --- | ------- | --- | --- | ---- | --- | -------- | --- |
| 0.8               |         |     | 0.8 |         | 0.8 |     |      | 0.8 |          |     |
| 0.6               |         |     | 0.6 |         | 0.6 |     |      | 0.6 |          |     |
| 0.4               |         |     | 0.4 |         | 0.4 |     |      | 0.4 |          |     |
| 0.2               |         |     | 0.2 |         | 0.2 |     |      | 0.2 |          |     |
| 0.0               |         |     | 0.0 |         | 0.0 |     |      | 0.0 |          |     |
|                   | 0.0 0.1 | 0.2 | 0.0 | 0.2 0.4 | 0.0 | 0.5 | 1.0  | 0.0 | 0.1      | 0.2 |
|                   | qmsum   |     |     | samsum  |     |     | trec |     | triviaqa |     |
| 1.0               |         |     | 1.0 |         | 1.0 |     |      | 1.0 |          |     |
erocs egarevA
| 0.8 |         |     | 0.8 |     | 0.8 |     |     | 0.8 |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |         |     |     |     | 0.6 |     |     | 0.6 |     |     |
| 0.6 |         |     | 0.6 |     |     |     |     |     |     |     |
| 0.4 |         |     | 0.4 |     | 0.4 |     |     | 0.4 |     |     |
| 0.2 |         |     | 0.2 |     | 0.2 |     |     | 0.2 |     |     |
| 0.0 |         |     | 0.0 |     | 0.0 |     |     | 0.0 |     |     |
|     | 0.0 0.2 | 0.4 | 0.0 | 0.2 | 0.0 | 0.1 | 0.2 | 0.0 | 0.2 | 0.4 |
Average TTFT (s) Average TTFT (s) Average TTFT (s) Average TTFT (s)
Figure12:EVICPRESS’sconsistentlyimprovesTTFT-qualitytrade-offondifferentdatasets.Amongthesedatasets,EVICPRESS
hasbetterimprovementonshortercontexts(suchasmulti_newsandqasper).
andquality.InsettingswithabundantDRAM(e.g.,models odsfromthreemethodsthatdropimportanttokensfromKV
withsmallKVcachesizeorworkloadswhereallKVcaches cache to shrink its memory footprint. EVICPRESS can be
canbekeptinfastmemory),theadvantageofsophisticated easilyextendedtochoosefromKVcachequantizationmeth-
jointoptimizationoversimplerpoliciesnaturallydiminishes. ods[16,17,32].Similarly,wecanbuildaprofilingconfigura-
Moreover,when the loading bandwidths for fast and slow tionpoolthatcombinesbothtokendroppingandquantization,
storage devices are similar, EVICPRESS’s benefits of joint andadaptsthefractionoftokenstobedroppedandthenum-
optimizationwilldecrease.Ourprototypeisevaluatedona berofbitsusedtorepresenteachKVelement.
single-nodeGPUsetup.ExtendingEVICPRESStomulti-node
deployments,cross-GPUcachesharing,highlymulti-tenant
9 Conclusion
environmentswithstrictisolationrequirements,andalterna-
tiveservingsystemsandstoragehierarchiesisanimportant
WepresentCacheServe,acachingsystemforlargelanguage
directionforfuturework.
modelinferenceworkloadthatco-designscompressionand
Adapting to changing loading bandwidth: EVICPRESS evictionpoliciesunderhierarchicalstoragedevicestoachieve
currentlyassumesthatthebandwidthofeachstoragetieris
betterquality-delaytradeoff.CacheServe’scoreinnovation
knownandrelativelystableoverthetimeatwhichweprofile liesinitscapabilitytojointlyexploitthebenefitofbothevic-
andre-profileconfigurations.Whentheavailablebandwidth
tionandcompression,bydefiningautilityfunctionandse-
changesmorerapidlythanourre-profilingfrequencyorwhen lectingthecompression-evictionthatmaximizessuchutility.
bandwidth fluctuations are strongly correlated with bursty WeimplementoursystemontopofproductionlevelLLM
queryarrivals,theprecomputedutilityfunctionscorescould
servingengine(vLLM)andcacheengine(LMCache).Our
temporarilydeviatefromthetruequality-latencytrade-offs evaluationsacrossdiversemodelsanddatasetsdemonstrate
andleadtosuboptimaldecisions.Aspartofourfuturework, thatCacheServesignificantlyreducesaverageTime-To-First-
weplantoextendEVICPRESStoreactmorequicklytoband-
Tokenby1.43to3.77×withequivalentqualityasbaselines
widthshifts,incorporatericheronlinetelemetry,andco-design orimprovesqualityby13.58%to55.40%atthesameTTFT.
bandwidth-awareprofilingandschedulingpolicies.
Bydesigningourforward-compatiblearchitectureforfuture
Extendingtoarichersetofcompressionmethods: Inthe KVcachecompressionandevictiontechniques,EVICPRESS
evaluation section, EVICPRESS selects compression meth- opensupanewdimensionforLLMinferenceoptimizations.
12

| References |     |     |     |     |     |     | Adaptcache:Kvcachenativestoragehierarchyforlow- |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- |
delayandhigh-qualitylanguagemodelserving,2025.
| [1] Amey | Agrawal, | Ashish | Panwar, | Jayashree |     | Mohan, |                                         |     |     |          |
| -------- | -------- | ------ | ------- | --------- | --- | ------ | --------------------------------------- | --- | --- | -------- |
|          |          |        |         |           |     |        | [12] HarshaRGaikwadandArvindWKiwelekar. |     |     | Agenera- |
NipunKwatra,BhargavS.Gulavani,andRamachandran
Ramjee. Sarathi:Efficientllminferencebypiggyback- tiveai-basedassistanttoevaluateshortandlonganswer
ingdecodeswithchunkedprefills,2023. questions. SNComputerScience,5(5):633,2024.
[2] Negar Arabzadeh, Amin Bigdeli, and Charles L. A. [13] BinGao,ZhuominHe,PuruSharma,QingxuanKang,
Clarke. Adaptingstandardretrievalbenchmarkstoeval- DjordjeJevdjic,JunboDeng,XingkunYang,ZhouYu,
uategeneratedanswers,2024. andPengfeiZuo. Attentionstore:Cost-effectiveatten-
tionreuseacrossmulti-turnconversationsinlargelan-
| [3] Yushi                                     | Bai,  | Xin Lv,    | Jiajie                | Zhang, | Hongchang | Lyu,      |                    |     |                                |     |
| --------------------------------------------- | ----- | ---------- | --------------------- | ------ | --------- | --------- | ------------------ | --- | ------------------------------ | --- |
|                                               |       |            |                       |        |           |           | guagemodelserving. |     | arXivpreprintarXiv:2403.19708, |     |
| JiankaiTang,ZhidianHuang,ZhengxiaoDu,XiaoLiu, |       |            |                       |        |           |           | 52:20–38,2024.     |     |                                |     |
| Aohan                                         | Zeng, | Lei Hou,   | Yuxiao                | Dong,  | Jie       | Tang, and |                    |     |                                |     |
| Juanzi                                        | Li.   | Longbench: | A bilingual,multitask |        |           | bench-    |                    |     |                                |     |
[14] InGim,GuojunChen,SeungseobLee,NikhilSarda,
mark for long context understanding. arXiv preprint Anurag Khandelwal, and Lin Zhong. Prompt cache:
arXiv:2308.14508,2023. Modularattentionreuseforlow-latencyinference,2023.
[4] ZiyiCao,QingyiSi,JingbinZhang,andBingquanLiu.
[15] ZhuohanGu,JiayiYao,KuntaiDu,andJunchenJiang.
Sparseattentionacrossmultiple-contextkvcache,2025.
Llmsteer:Improvinglong-contextllminferencebysteer-
ingattentiononreusedcontexts,2024.
| [5] Weijian | Chen,  | Shuibing | He,        | Haoyang | Qu,    | Ruidong |     |     |     |     |
| ----------- | ------ | -------- | ---------- | ------- | ------ | ------- | --- | --- | --- | --- |
| Zhang,      | Siling | Yang,    | Ping Chen, | Yi      | Zheng, | Baoxing |     |     |     |     |
[16] YefeiHe,LuomingZhang,WeijiaWu,JingLiu,Hong
| Huai,andGangChen. |     |     | {IMPRESS}:An{Importance- |     |     |     |                      |     |                           |     |
| ----------------- | --- | --- | ------------------------ | --- | --- | --- | -------------------- | --- | ------------------------- | --- |
|                   |     |     |                          |     |     |     | Zhou,andBohanZhuang. |     | Zipcache:Accurateandeffi- |     |
Informed}{Multi-Tier}prefix{KV}storagesystemfor cientkvcachequantizationwithsalienttokenidentifica-
| largelanguagemodelinference. |     |     |     | In23rdUSENIXCon- |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- |
tion,2024.
| ference | on  | File and | Storage | Technologies |     | (FAST 25), |     |     |     |     |
| ------- | --- | -------- | ------- | ------------ | --- | ---------- | --- | --- | --- | --- |
pages187–201,2025. [17] ColemanHooper,SehoonKim,HivaMohammadzadeh,
MichaelWMahoney,YakunSophiaShao,KurtKeutzer,
[6] YilongChen,GuoxiaWang,JunyuanShang,ShiyaoCui,
|     |     |     |     |     |     |     | andAmirGholami. | Kvquant:Towards10millioncon- |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ---------------------------- | --- | --- |
ZhenyuZhang,TingwenLiu,ShuohuanWang,YuSun,
|                     |     |     |                           |     |     |     | text length | llm inference | with kv cache | quantization. |
| ------------------- | --- | --- | ------------------------- | --- | --- | --- | ----------- | ------------- | ------------- | ------------- |
| DianhaiYu,andHuaWu. |     |     | Nacl:Ageneralandeffective |     |     |     |             |               |               |               |
arXivpreprintarXiv:2401.18079,2024.
kvcacheevictionframeworkforllmsatinferencetime,
2024.
[18] BoJiang,TaolueYang,YouyuanLiu,ChengmingZhang,
|     |     |     |     |     |     |     | XubinHe,andSianJin. |     | Kvcomp:Ahigh-performance, |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ------------------------- | --- |
[7] WenCheng,ShichenDong,JiayuQin,andWeiWang.
llm-aware,lossycompressionframeworkforkvcache,
| Qaq:Qualityadaptivequantizationforllmkvcache. |     |     |     |     |     | In  |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2025.
ProceedingsoftheIEEE/CVFInternationalConference
onComputerVision,pages2542–2550,2025.
|     |     |     |     |     |     |     | [19] Huiqiang | Jiang, Qianhui | Wu, Chin-Yew | Lin, Yuqing |
| --- | --- | --- | --- | --- | --- | --- | ------------- | -------------- | ------------ | ----------- |
[8] WeihaoCui,YukangChen,HanZhao,ZiyiXu,Quan Yang,andLiliQiu. Llmlingua:Compressingprompts
|     |     |     |     |     |     |     | for accelerated | inference | of large language | models. |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --------- | ----------------- | ------- |
Chen,XushengChen,YangjieZhou,ShixuanSun,and
arXivpreprintarXiv:2310.05736,2023.
| MinyiGuo. |     | Optimizingslo-orientedllmservingwith |     |     |     |     |     |     |     |     |
| --------- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
pd-multiplexing,2025.
[20] HuiqiangJiang,QianhuiWu,XufangLuo,Dongsheng
[9] Alessio Devoto, Yu Zhao, Simone Scardapane, and Li,Chin-YewLin,YuqingYang,andLiliQiu. Longllm-
lingua:Acceleratingandenhancingllmsinlongcontext
| Pasquale | Minervini. |     | A simple | and | effective | l norm- |     |     |     |     |
| -------- | ---------- | --- | -------- | --- | --------- | ------- | --- | --- | --- | --- |
2
basedstrategyforkvcachecompression,2024. scenariosviapromptcompression,2023.
[10] Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel [21] JunchenJiang,YuhanLiu,ZhuohanGu,QizhengZhang,
ChenWang,YueZhu,RuoyuQin,YuweiAn,Xiangfeng
| Guha, | Dylan | Zinsley, | Emily | Liu, Will | Tennien, | Atri |     |     |     |     |
| ----- | ----- | -------- | ----- | --------- | -------- | ---- | --- | --- | --- | --- |
Rudra,JamesZou,AzaliaMirhoseini,andChristopher Zhu,KuntaiDu,etal. Networkingforstatefulllminfer-
Re. Cartridges:Lightweightandgeneral-purposelong ence. InConferenceoftheACMSpecialInterestGroup
onDataCommunication,2025.
contextrepresentationsviaself-study,2025.
[11] ShaotingFeng,HanchenLi,KuntaiDu,ZhuohanGu, [22] ShuoweiJin,XueshenLiu,QingzhaoZhang,andZ.Mor-
YuhanLiu,JiayiYao,SiddhantRay,SamuelShen,Yihua ley Mao. Compute orloadkv cache? why notboth?,
| Cheng,GaneshAnanthanarayanan,andJunchenJiang. |     |     |     |     |     |     | 2025. |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- |
13

[23] Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Jia. SpecInfer:AcceleratingGenerativeLLMServing
Lee,SangdooYun,andHyunOhSong. Kvzip:Query- withSpeculativeInferenceandTokenTreeVerification.
agnostickvcachecompressionwithcontextreconstruc- arXivpreprintarXiv:2305.09781,2023.
tion,2025.
[34] Xiurui Pan, Endian Li, Qiao Li, Shengwen Liang,
[24] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Yizhou Shan,Ke Zhou,Yingwei Luo,Xiaolin Wang,
Sheng,LianminZheng,CodyHaoYu,JosephE.Gonza- andJieZhang. Instinfer:In-storageattentionoffloading
lez,HaoZhang,andIonStoica. Efficientmemoryman- forcost-effectivelong-contextllminference,2024.
agementforlargelanguagemodelservingwithpagedat-
[35] Junyoung Park, Dalton Jones, Matthew J Morse,
tention,2023.
RaghavvGoel,MinguLee,andChrisLott.Keydiff:Key
[25] YanivLeviathan,MatanKalman,andY.Matias. FastIn- similarity-basedkvcacheevictionforlong-contextllm
ferencefromTransformersviaSpeculativeDecoding.In inferenceinresource-constrainedenvironments,2025.
InternationalConferenceonMachineLearning,2022.
[36] PratyushPatel,EshaChoukse,ChaojieZhang,Aashaka
[26] Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Shah,ÍñigoGoiri,SaeedMaleki,andRicardoBianchini.
Zhang,Huanzhi Mao,Xiaokun Chen,Alvin Cheung, Splitwise:Efficientgenerativellminferenceusingphase
JosephGonzalez,andIonStoica. Continuum:Efficient splitting,2024.
androbustmulti-turnllmagentschedulingwithkvcache
[37] RuoyuQin,ZhemingLi,WeiranHe,MingxingZhang,
time-to-live,2025.
YongweiWu,WeiminZheng,andXinranXu. Moon-
[27] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat cake:Akvcache-centricdisaggregatedarchitecturefor
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, llmserving,2025.
PatrickLewis,andDemingChen. Snapkv:Llmknows
[38] ZiranQin,YuchenCao,MingbaoLin,WenHu,Shixuan
whatyouarelookingforbeforegeneration,2024.
Fan, Ke Cheng, Weiyao Lin, and Jianguo Li. Cake:
[28] AkideLiu,JingLiu,ZizhengPan,YefeiHe,Gholamreza Cascading and adaptive kv cache eviction with layer
Haffari,andBohanZhuang. Minicache:Kvcachecom- preferences,2025.
pressionindepthdimensionforlargelanguagemodels,
[39] SiddhantRay,RuiPan,ZhuohanGu,KuntaiDu,Shaot-
2024.
ing Feng,Ganesh Ananthanarayanan,Ravi Netravali,
[29] YuhanLiu,YihuaCheng,JiayiYao,YuweiAn,Xiaokun andJunchenJiang. Metis:Fastquality-awareragsys-
Chen,ShaotingFeng,YuyangHuang,SamuelShen,Rui temswithconfigurationadaptation. InProceedingsof
Zhang,KuntaiDu,andJunchenJiang. Lmcache:Anef- theACMSIGOPS31stSymposiumonOperatingSys-
ficientkvcachelayerforenterprise-scalellminference, temsPrinciples,pages606–622,2025.
2025.
[40] NilsReimersandIrynaGurevych. Sentence-bert:Sen-
[30] YuhanLiu,YuyangHuang,JiayiYao,ShaotingFeng, tenceembeddingsusingsiamesebert-networks. InPro-
Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, ceedingsofthe2019ConferenceonEmpiricalMethods
JunchenJiang,ShanLu,MadanMusuvathi,andEsha inNaturalLanguageProcessing.AssociationforCom-
Choukse. Droidspeak:Kvcachesharingforcross-llm putationalLinguistics,112019.
communicationandmulti-llmserving,2025.
[41] MohammadShahrad,RodrigoFonseca,InigoGoiri,Go-
[31] Yuhan Liu,Hanchen Li,Kuntai Du,Jiayi Yao,Yihua harChaudhry,PaulBatum,JasonCooke,EduardoLau-
Cheng,YuyangHuang,ShanLu,MichaelMaire,Henry reano,ColbyTresness,MarkRussinovich,andRicardo
Hoffmann,AriHoltzman,etal. Cachegen:Fastcontext Bianchini. Serverlessinthewild:Characterizingand
loadingforlanguagemodelapplications. arXivpreprint optimizing the serverless workload at a large cloud
arXiv:2310.07240,2023. provider. In2020USENIXannualtechnicalconference
(USENIXATC20),pages205–218,2020.
[32] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong,
Zhaozhuo Xu,Vladimir Braverman,Beidi Chen,and [42] WoominSong,SeunghyukOh,SangwooMo,Jaehyung
Xia Hu. Kivi: A tuning-free asymmetric 2bit quanti- Kim,SukminYun,Jung-WooHa,andJinwooShin. Hi-
zationforkvcache. arXivpreprintarXiv:2402.02750, erarchicalcontextmerging:Betterlongcontextunder-
2024. standingforpre-trainedllms,2024.
[33] XupengMiao,GabrieleOliaro,ZhihaoZhang,Xinhao [43] VikranthSrivatsa,ZijianHe,ReynaAbhyankar,Dong-
Cheng,Zeyu Wang,Rae Ying Yee Wong,Zhuoming mingLi,andYiyingZhang. Preble:Efficientdistributed
Chen,DaiyaanArfeen,ReynaAbhyankar,andZhihao promptschedulingforllmserving,2024.
14

[44] DezhanTu,DanyloVashchilenko,YuzheLu,andPan- [54] Jiayi Yao,Hanchen Li,Yuhan Liu,Siddhant Ray,Yi-
pan Xu. Vl-cache: Sparsity and modality-aware kv huaCheng,QizhengZhang,KuntaiDu,ShanLu,and
cachecompressionforvision-languagemodelinference JunchenJiang. Cacheblend:Fastlargelanguagemodel
acceleration,2024. servingforragwithcachedknowledgefusion,2025.
[45] ZhongweiWan,ZiangWu,CheLiu,JinfaHuang,Zhi- [55] Bohan Yu andYekun Chai. Evolkv: Evolutionarykv
hongZhu,PengJin,LongyueWang,andLiYuan. Look- cachecompressionforllminference,2025.
| m: Look-once |     | optimization | in kv | cache | for | efficient |     |     |     |     |     |     |     |
| ------------ | --- | ------------ | ----- | ----- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
multimodallong-contextinference,2024. [56] Huawei Zhang, Chunwei Xia, and Zheng Wang.
|     |     |     |     |     |     |     | Kvswap: |     | Disk-aware | kv  | cache | offloading | for long- |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | ---------- | --- | ----- | ---------- | --------- |
[46] YuxinWang,YuhanChen,ZeyuLi,XuezeKang,Yuchu
contexton-deviceinference,2025.
Fang,YejuZhou,YangZheng,ZhenhengTang,XinHe,
| Rui Guo,Xin |     | Wang,Qiang | Wang,Amelie |     | Chi | Zhou, |              |              |     |                |     |     |           |
| ----------- | --- | ---------- | ----------- | --- | --- | ----- | ------------ | ------------ | --- | -------------- | --- | --- | --------- |
|             |     |            |             |     |     |       | [57] Xinrong | Zhang,Yingfa |     | Chen,Shengding |     |     | Hu,Zihang |
and Xiaowen Chu. Burstgpt: A real-world workload Xu,JunhaoChen,MooKhaiHao,XuHan,ZhenLeng
datasettooptimizellmservingsystems,2025. Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun.
|     |     |     |     |     |     |     | ∞bench: | Extending |     | long | context | evaluation | beyond |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | ---- | ------- | ---------- | ------ |
[47] Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and 100ktokens,2024.
| QuocV.Le. | Simplesyntheticdatareducessycophancy |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inlargelanguagemodels,2024.
|     |     |     |     |     |     |     | [58] YanqiZhang,ÍñigoGoiri,GoharIrfanChaudhry,Ro- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
drigoFonseca,SamehElnikety,ChristinaDelimitrou,
[48] JerryWei,ChengrunYang,XinyingSong,YifengLu,
|     |     |     |     |     |     |     | andRicardoBianchini. |     |     | Fasterandcheaperserverless |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | -------------------------- | --- | --- | --- |
NathanHu,JieHuang,DustinTran,DaiyiPeng,Ruibo
computingonharvestedresources.InProceedingsofthe
Liu,DaHuang,CosmoDu,andQuocV.Le. Long-form ACMSIGOPS28thSymposiumonOperatingSystems
factualityinlargelanguagemodels,2024.
Principles,pages724–739,2021.
[49] ColinWhite,SamuelDooley,ManleyRoberts,ArkaPal,
|                      |     |     |            |                  |     |     | [59] Zhenyu | Zhang, | Ying | Sheng, | Tianyi | Zhou, | Tianlong |
| -------------------- | --- | --- | ---------- | ---------------- | --- | --- | ----------- | ------ | ---- | ------ | ------ | ----- | -------- |
| Ben Feuer,Siddhartha |     |     | Jain,Ravid | Shwartz-Ziv,Neel |     |     |             |        |      |        |        |       |          |
Chen,LianminZheng,RuisiCai,ZhaoSong,Yuandong
Jain,KhalidSaifullah,SreemantiDey,Shubh-Agrawal,
|         |       |         |           |        |         |     | Tian,ChristopherRé,ClarkBarrett,etal. |     |     |     |     |     | H2o:Heavy- |
| ------- | ----- | ------- | --------- | ------ | ------- | --- | ------------------------------------- | --- | --- | --- | --- | --- | ---------- |
| Sandeep | Singh | Sandha, | Siddartha | Naidu, | Chinmay |     |                                       |     |     |     |     |     |            |
hitteroracleforefficientgenerativeinferenceoflarge
Hegde,YannLeCun,TomGoldstein,WillieNeiswanger,
|           |           |     |            |     |              |     | languagemodels. |     |     | AdvancesinNeuralInformationPro- |     |     |     |
| --------- | --------- | --- | ---------- | --- | ------------ | --- | --------------- | --- | --- | ------------------------------- | --- | --- | --- |
| and Micah | Goldblum. |     | Livebench: | A   | challenging, |     |                 |     |     |                                 |     |     |     |
cessingSystems,36,2024.
contamination-limitedllmbenchmark,2025.
|               |      |      |          |       |       |     | [60] Lianmin | Zheng, | Liangsheng |     | Yin, | Zhiqiang | Xie, Jeff |
| ------------- | ---- | ---- | -------- | ----- | ----- | --- | ------------ | ------ | ---------- | --- | ---- | -------- | --------- |
| [50] Zhiqiang | Xie, | Ziyi | Xu, Mark | Zhao, | Yuwei | An, |              |        |            |     |      |          |           |
Huang,ChuyueSun,CodyHaoYu,ShiyiCao,Chris-
VikramSharmaMailthody,ScottMahlke,MichaelGar-
|                            |     |     |                         |     |     |     | tos | Kozyrakis, | Ion | Stoica, | Joseph | E Gonzalez, | et al. |
| -------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | ---------- | --- | ------- | ------ | ----------- | ------ |
| land,andChristosKozyrakis. |     |     | Strata:Hierarchicalcon- |     |     |     |     |            |     |         |        |             |        |
Efficientlyprogramminglargelanguagemodelsusing
textcachingforlongcontextlanguagemodelserving.
|     |     |     |     |     |     |     | sglang. | arXivpreprintarXiv:2312.07104,2023. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----------------------------------- | --- | --- | --- | --- | --- |
arXivpreprintarXiv:2508.18572,2025.
|     |     |     |     |     |     |     | [61] YinminZhong,ShengyuLiu,JundaChen,JianboHu, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
[51] JingXiong,JianghanShen,FanghuaYe,ChaofanTao,
|          |      |          |         |     |           |     | YiboZhu,XuanzheLiu,XinJin,andHaoZhang. |     |     |     |     |     | Dist- |
| -------- | ---- | -------- | ------- | --- | --------- | --- | -------------------------------------- | --- | --- | --- | --- | --- | ----- |
| Zhongwei | Wan, | Jianqiao | Lu, Xun | Wu, | Chuanyang |     |                                        |     |     |     |     |     |       |
serve:Disaggregatingprefillanddecodingforgoodput-
Zheng,ZhijiangGuo,MinYang,LingpengKong,and
|           |                                     |     |     |     |     |     | optimizedlargelanguagemodelserving. |     |     |     |     |     | arXivpreprint |
| --------- | ----------------------------------- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | ------------- |
| NgaiWong. | Uncomp:Canmatrixentropyuncoverspar- |     |     |     |     |     |                                     |     |     |     |     |     |               |
arXiv:2401.09670,2024.
sity?–acompressordesignfromanuncertainty-aware
perspective,2025.
|     |     |     |     |     |     |     | [62] XiabinZhou,WenbinWang,MinyanZeng,JiaxianGuo, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
[52] YiXiong,HaoWu,ChangxuShao,ZiqingWang,Rui XueboLiu,LiShen,MinZhang,andLiangDing. Dy-
|        |        |      |               |     |        |     | namickv: |     | Task-aware | adaptive | kv  | cache | compression |
| ------ | ------ | ---- | ------------- | --- | ------ | --- | -------- | --- | ---------- | -------- | --- | ----- | ----------- |
| Zhang, | Yuhong | Guo, | Junping Zhao, | Ke  | Zhang, | and |          |     |            |          |     |       |             |
forlongcontextllms,2025.
| Zhenxuan | Pan. | Layerkv: | Optimizing |     | large language |     |     |     |     |     |     |     |     |
| -------- | ---- | -------- | ---------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
modelservingwithlayer-wisekvcachemanagement,
|     |     |     |     |     |     |     | [63] JianianZhu,HangWu,HaojieWang,YinghuiLi,Biao |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
2024.
|     |     |     |     |     |     |     | Hou,Ruixuan |     | Li,and | Jidong | Zhai. | Fastcache: | Opti- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | ------ | ----- | ---------- | ----- |
mizingmultimodalllmservingthroughlightweightkv-
[53] JingboYang,BairuHou,WeiWei,YujiaBao,andShiyu
cachecompressionframework,2025.
| Chang. | Kvlink:Acceleratinglargelanguagemodelsvia |     |     |     |     |     |     |     |     |     |     |     |     |
| ------ | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
efficientkvcachereuse,2025.
15

A PseudoCodeforAlgorithm
Algorithm1:PolicyOptimizer
Input: IncomingKVCachethatneedstobestoredi(sizeof
thekvcachel,qualityfunctionQ(compressionratio)),
i i
StoragequeueT,CurrentOccupiedsizeS ,Storage
T
memorysizeC
Output: ListofoperationsonKVCache:[(KVCachei,
compressionratior,operationo)]
i i
1: r i ←argmaxQ i (r)
2: T.append(i,r i )
3: L.append((i,r i ,insert))
4: S T =S T +sizeof(l i ∗r i )
5: whileS T >Cdo
6: j,r,o,←KVCacheinT withtheleastutilitydrop
operationandtheoperation.
7: ifo==compressthen
8: S=S T −T j ∗(r j −r)
9: elseifo==evict then
10: r←0
11: S=S T −(T j ∗r j )
12: endif
13: L.append((j,r,o))
14: endwhile
15: return Ltoexecutor.
16