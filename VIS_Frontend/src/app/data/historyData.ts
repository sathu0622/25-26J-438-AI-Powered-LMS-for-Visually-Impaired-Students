// Sri Lankan History Curriculum Data

export interface Topic {
  id: number;
  title: string;
  content: string;
  duration: string; // in seconds for audio playback
}

export interface Lesson {
  id: number;
  title: string;
  description: string;
  duration: string;
  topics: Topic[];
}

export const grade10Lessons: Lesson[] = [
  {
    id: 1,
    title: 'Prehistoric & Protohistoric Sri Lanka',
    description: 'Discover the earliest inhabitants of Sri Lanka, the Balangoda Man, and archaeological evidence',
    duration: '25 min',
    topics: [
      {
        id: 1,
        title: 'Prehistoric Man (Balangoda Man)',
        content: `Balangoda Man represents the earliest known inhabitants of Sri Lanka, dating back to approximately 37,000 years ago. Named after the town of Balangoda where their remains were first discovered, these prehistoric people were anatomically modern humans who adapted to the tropical environment of ancient Sri Lanka. They were characterized by their robust build and had developed tools and survival strategies suited to the island's climate and geography. Archaeological evidence suggests they lived in caves and rock shelters, primarily in the wet zone of Sri Lanka.`,
        duration: '120',
      },
      {
        id: 2,
        title: 'Tools, Lifestyle, and Food Habits',
        content: `The Balangoda Man used stone tools made from quartz and chert, demonstrating advanced craftsmanship for their time. These tools included hand axes, scrapers, and arrow points used for hunting and food preparation. Their lifestyle was semi-nomadic, moving between seasonal camps. They were hunter-gatherers who relied on wild animals, particularly deer and wild boar, as well as fruits, roots, and edible plants. Evidence suggests they also collected honey and shellfish from coastal areas. Their diet was diverse and adapted to the tropical environment, showing sophisticated knowledge of local flora and fauna.`,
        duration: '130',
      },
      {
        id: 3,
        title: 'Archaeological Evidence (Belilena, Pahiyangala)',
        content: `The Belilena Cave in Kitulgala and Pahiyangala Cave near Bulathsinhala are two of the most significant archaeological sites revealing the life of Balangoda Man. At Belilena Cave, archaeologists discovered human skeletal remains, stone tools, and evidence of fire use dating back 32,000 years. The cave also revealed ornamental beads made from shells, indicating early aesthetic sense and possibly trade. Pahiyangala Cave, one of the largest natural rock formations in Asia, provided evidence of continuous human occupation spanning thousands of years. Excavations revealed stratified layers showing technological progression and adaptation to environmental changes over millennia.`,
        duration: '150',
      },
    ],
  },
  {
    id: 2,
    title: 'Early Settlements & Civilization',
    description: 'Learn about the arrival of Aryans and the development of early agricultural communities',
    duration: '20 min',
    topics: [
      {
        id: 4,
        title: 'Arrival of Aryans',
        content: `The arrival of Indo-Aryan settlers from North India, traditionally dated around 543 BCE according to the Mahavamsa chronicle, marked a transformative period in Sri Lankan history. These migrants, led by Prince Vijaya according to legend, brought with them advanced agricultural techniques, social organization systems, and the Prakrit language which evolved into Sinhala. While the exact date and circumstances remain subjects of scholarly debate, linguistic and archaeological evidence confirms significant cultural influence from the Indian subcontinent during this period. The Aryans established the first organized settlements and introduced concepts of governance and social hierarchy.`,
        duration: '140',
      },
      {
        id: 5,
        title: 'Development of Villages',
        content: `The development of permanent villages marked the transition from nomadic to settled life in ancient Sri Lanka. Villages were typically established near water sources and fertile land suitable for cultivation. The basic village structure included residential areas, agricultural fields, and communal spaces. Villages were governed by local chiefs or headmen who maintained order and resolved disputes. The social structure within villages was hierarchical, with distinctions based on occupation and family lineage. Trade between villages facilitated the exchange of goods and ideas, leading to cultural development and economic specialization.`,
        duration: '130',
      },
      {
        id: 6,
        title: 'Agriculture & Irrigation Beginnings',
        content: `The introduction of systematic agriculture revolutionized Sri Lankan society around the 4th century BCE. Rice cultivation became the primary agricultural activity, suited to the island's climate and terrain. Early farmers developed innovative irrigation techniques to manage water resources, particularly in the dry zone. Simple water conservation methods like bunding fields and creating small reservoirs evolved into more complex systems. The construction of small tanks (wewa) for water storage marked the beginning of Sri Lanka's renowned hydraulic civilization. These early irrigation works laid the foundation for the sophisticated tank and canal systems that would later define Anuradhapura and Polonnaruwa periods.`,
        duration: '145',
      },
    ],
  },
  {
    id: 3,
    title: 'Anuradhapura Kingdom',
    description: 'Explore the golden age of Sri Lankan civilization with great kings and Buddhist culture',
    duration: '35 min',
    topics: [
      {
        id: 7,
        title: 'Founding by King Pandukabhaya',
        content: `King Pandukabhaya founded Anuradhapura around 437 BCE, establishing it as the first major capital of ancient Sri Lanka. According to the Mahavamsa, Pandukabhaya was a remarkable town planner who organized the city with designated areas for different communities and professions. He created public facilities including hospitals, parks, and cemeteries. The city was laid out with broad streets and a sophisticated drainage system. Pandukabhaya established a city council and administrative structure that would serve as a model for future rulers. His 70-year reign brought stability and prosperity, setting the stage for Anuradhapura to become one of the greatest cities of the ancient world.`,
        duration: '145',
      },
      {
        id: 8,
        title: 'King Devanampiyatissa',
        content: `King Devanampiyatissa (307-267 BCE) holds immense significance as the ruler who introduced Buddhism to Sri Lanka. His friendship with the Indian Emperor Ashoka led to the arrival of Arahant Mahinda, Ashoka's son, who brought Buddhist teachings to the island. Devanampiyatissa embraced Buddhism and sponsored the construction of the Mahavihara monastery, one of the most important Buddhist institutions in Sri Lankan history. He also facilitated the arrival of Sanghamitta Theri, who brought a sapling of the sacred Bodhi tree under which Buddha attained enlightenment. This sapling, planted at Anuradhapura, still stands today as the oldest historically authenticated tree in the world.`,
        duration: '150',
      },
      {
        id: 9,
        title: 'King Dutugemunu',
        content: `King Dutugemunu (161-137 BCE) is celebrated as one of Sri Lanka's greatest warrior kings. He unified the island by defeating the South Indian king Elara, who had ruled the northern part of Sri Lanka for 44 years. Dutugemunu's campaign to reclaim Anuradhapura is chronicled in the Mahavamsa with epic detail. After his victory, he embarked on an ambitious building program, constructing the magnificent Ruwanwelisaya stupa, one of the most sacred Buddhist monuments. The Mirisawetiya stupa and the Brazen Palace (Lovamahapaya) were also built during his reign. Dutugemunu is remembered not just as a warrior but as a devoted Buddhist who balanced military prowess with religious devotion.`,
        duration: '155',
      },
      {
        id: 10,
        title: 'King Valagamba',
        content: `King Valagamba (89-77 BCE) faced significant challenges during his reign, including invasions by South Indian forces that temporarily drove him from the throne. During his 14 years in hiding, he vowed to build a great monastery if he regained power. Upon his restoration, he fulfilled this vow by constructing the Abhayagiri monastery, which became one of the three main monastic fraternities in Anuradhapura. Valagamba also sponsored the first written compilation of the Pali Buddhist canon in Sri Lanka, a monumental achievement that preserved Buddhist teachings for future generations. His reign demonstrated resilience and religious dedication despite political turmoil.`,
        duration: '140',
      },
      {
        id: 11,
        title: 'Introduction of Buddhism',
        content: `The introduction of Buddhism in the 3rd century BCE fundamentally transformed Sri Lankan civilization. Buddhism became the state religion and influenced every aspect of society - art, architecture, literature, education, and governance. Monasteries became centers of learning where monks preserved and transmitted knowledge. The Buddhist concept of righteousness (dhamma) influenced legal and administrative systems. The religion promoted values of compassion, non-violence, and social welfare, leading to the establishment of hospitals and charitable institutions. Buddhism also fostered literary development, with extensive Pali and Sinhala Buddhist literature. The religion's emphasis on education led to high literacy rates among monks and the aristocracy.`,
        duration: '150',
      },
      {
        id: 12,
        title: 'Irrigation Systems (Tanks & Canals)',
        content: `The Anuradhapura period witnessed the development of one of the ancient world's most sophisticated irrigation systems. Kings and engineers built massive tanks (reservoirs) to collect and store rainwater, enabling agriculture in the dry zone. The systems included interconnected tanks linked by canals, allowing water transfer across long distances. Notable constructions include the Abhaya wewa built by King Pandukabhaya, the Basawakkulama tank, and later the giant Tissa wewa. These irrigation works demonstrated advanced hydraulic engineering, including precise gradient calculations for canal systems and spillway designs. The tanks also served multiple purposes - irrigation, drinking water supply, and beautification of the landscape.`,
        duration: '145',
      },
      {
        id: 13,
        title: 'Administration, Economy & Society',
        content: `The Anuradhapura Kingdom developed a sophisticated administrative system with the king at the apex, supported by ministers, provincial governors, and village headmen. The economy was primarily agricultural, with rice as the staple crop. Trade flourished both domestically and internationally, with Sri Lankan merchants trading gems, pearls, elephants, and spices with Rome, China, and Southeast Asia. Society was hierarchically organized with distinct classes - the aristocracy, merchants, farmers, and artisans. Buddhism promoted social welfare, and kings built hospitals (excluding none) and rest houses for travelers. The society valued education, with pirivenas (monastic schools) providing Buddhist and secular education. Women had certain rights, including property ownership, though society was patriarchal overall.`,
        duration: '160',
      },
    ],
  },
  {
    id: 4,
    title: 'Polonnaruwa Kingdom',
    description: 'Study the medieval capital and its magnificent irrigation achievements',
    duration: '28 min',
    topics: [
      {
        id: 14,
        title: 'Rise of Polonnaruwa',
        content: `Polonnaruwa rose to prominence in the 11th century CE when it became necessary to establish a new capital after Anuradhapura faced repeated South Indian invasions. The Chola dynasty from South India conquered Anuradhapura in 993 CE and established Polonnaruwa as their administrative center. The city's strategic location in the central dry zone, protected by the Mahaweli River and surrounded by defensive walls, made it more secure than Anuradhapura. When Sri Lankan kings regained independence, they retained Polonnaruwa as the capital, recognizing its advantages. The city would flourish for nearly 200 years as the center of Sri Lankan civilization.`,
        duration: '140',
      },
      {
        id: 15,
        title: 'King Vijayabahu I',
        content: `King Vijayabahu I (1055-1110 CE) was the liberator who expelled the Chola invaders and restored Sinhalese rule after 75 years of foreign occupation. His 17-year struggle to regain control of the island demonstrated remarkable military strategy and perseverance. After victory, Vijayabahu I focused on rebuilding the war-torn nation. He restored Buddhist institutions that had deteriorated under Chola rule, including re-establishing the higher ordination of monks by bringing monks from Burma. He repaired irrigation systems damaged during the conflicts and promoted agricultural revival. Vijayabahu I established Polonnaruwa as the permanent capital, initiating construction projects that his successors would expand.`,
        duration: '145',
      },
      {
        id: 16,
        title: 'King Parakramabahu I',
        content: `King Parakramabahu I (1153-1186 CE) is considered the greatest ruler of Polonnaruwa and one of Sri Lanka's most accomplished monarchs. His famous dictum, "Let not even a drop of water that comes from the rain go to the sea without benefiting man," exemplifies his commitment to development. He unified the three kingdoms that had divided Sri Lanka and brought unprecedented prosperity. Parakramabahu I was a great patron of Buddhism, reforming the sangha and unifying different Buddhist sects. He built numerous temples and stupas, with the Rankot Vehera being the largest. His reign saw flourishing arts, literature, and architecture. He also maintained a strong military, even sending expeditions to South India and Burma.`,
        duration: '160',
      },
      {
        id: 17,
        title: 'Irrigation (Parakrama Samudra)',
        content: `The Parakrama Samudra (Sea of Parakrama) stands as the crowning achievement of ancient Sri Lankan irrigation engineering. Built by King Parakramabahu I, this massive reservoir covers approximately 5,940 acres and originally consisted of three separate tanks that were later interconnected. The tank could irrigate over 18,000 acres of paddy fields. Its construction required precise engineering calculations, massive labor organization, and innovative techniques to create embankments strong enough to hold vast quantities of water. The Parakrama Samudra not only served agricultural needs but also enhanced Polonnaruwa's beauty and strategic water security. Today it remains functional and stands as a testament to medieval Sri Lankan technological advancement.`,
        duration: '145',
      },
      {
        id: 18,
        title: 'Decline of Polonnaruwa',
        content: `Polonnaruwa's decline began in the late 13th century due to multiple factors. After Parakramabahu I's death, succession disputes weakened central authority. Repeated invasions from South India, particularly by the Pandyan dynasty, destabilized the kingdom. The invasion by Kalinga Magha in 1215 CE proved devastating, bringing destruction to Buddhist institutions and irrigation systems. Malaria, possibly spreading due to disrupted irrigation maintenance, made the dry zone increasingly uninhabitable. Economic decline followed as trade disrupted and agricultural productivity fell. By the late 13th century, Sri Lankan kingdoms shifted to the southwest wet zone, abandoning Polonnaruwa. The great city fell into ruin, reclaimed by jungle until its rediscovery and restoration in modern times.`,
        duration: '150',
      },
    ],
  },
  {
    id: 5,
    title: 'Cultural Achievements of Ancient Sri Lanka',
    description: 'Discover the artistic, literary, and architectural masterpieces of ancient civilization',
    duration: '22 min',
    topics: [
      {
        id: 19,
        title: 'Buddhist Literature',
        content: `Ancient Sri Lanka made extraordinary contributions to Buddhist literature. The most significant achievement was the writing down of the Tripitaka (Buddhist canon) in Pali during the reign of King Valagamba in the 1st century BCE, preserving Buddha's teachings for posterity. The Mahavamsa, written by Mahanama Thera in the 6th century CE, is a historical chronicle of exceptional literary merit. Commentaries on Buddhist texts, such as those by Buddhaghosa, were composed in Sri Lanka. Sinhala literature flourished with works like the Sikhāvalaya and Muvadevdāvata. Poetic works describing pilgrimage sites and Buddhist teachings demonstrated sophisticated literary development. These texts were copied and preserved in monasteries, creating a strong textual tradition.`,
        duration: '150',
      },
      {
        id: 20,
        title: 'Art & Architecture',
        content: `Ancient Sri Lankan art and architecture achieved remarkable sophistication. Stupas (dagobas) like the Ruwanwelisaya, Jetavanaramaya, and Abhayagiri reached enormous proportions, ranking among the tallest structures of the ancient world. Architectural techniques included precise mathematical calculations and innovative construction methods. The Lovamahapaya (Brazen Palace) featured nine stories with a bronze-tiled roof. Moonstones (sandakada pahana) carved at entrances showed intricate designs symbolizing the path to enlightenment. Guard stones featured divine figures. Rock temples showcased impressive integration of architecture with natural features. Polonnaruwa's architectural marvels, including the Gal Vihara rock sculptures and the Vatadage, demonstrated evolving styles and technical excellence.`,
        duration: '155',
      },
      {
        id: 21,
        title: 'Sculpture & Painting',
        content: `Sri Lankan sculptors created masterpieces that rank among the finest in Buddhist art. The Gal Vihara in Polonnaruwa features four magnificent Buddha statues carved from granite, including a 14-meter reclining Buddha of extraordinary grace and proportion. The Samadhi Buddha statue at Anuradhapura, carved in the 4th century CE, exemplifies serene meditation. Moonstone carvings showed sophisticated understanding of symbolism and aesthetics. Frescoes at Sigiriya, dating to the 5th century CE, demonstrate advanced painting techniques with vibrant mineral-based pigments. The paintings depict celestial maidens with grace and beauty, showing sophisticated understanding of human anatomy and movement. Temple wall paintings illustrated Buddhist Jataka tales with narrative skill and artistic expression.`,
        duration: '150',
      },
      {
        id: 22,
        title: 'Writing Systems & Inscriptions',
        content: `The development of writing systems in Sri Lanka provides crucial evidence of cultural and administrative sophistication. The Brahmi script was adapted for writing Prakrit and early Sinhala from around the 3rd century BCE. Rock inscriptions, particularly those recording donations to Buddhist monasteries, provide historical records of social organization and religious practices. The earliest inscriptions are simple, but over time they became more elaborate, recording detailed information about grants, boundaries, and privileges. By the medieval period, Sinhala script had evolved its distinctive characteristics. Inscriptions were carved on rock surfaces, stone pillars, and metal plates, demonstrating literacy among the ruling class and administrative efficiency in record-keeping.`,
        duration: '145',
      },
    ],
  },
];

export const grade11Lessons: Lesson[] = [
  {
    id: 6,
    title: 'Transitional Period Kingdoms',
    description: 'Examine the shift from dry zone to wet zone and the emergence of new capitals',
    duration: '30 min',
    topics: [
      {
        id: 23,
        title: 'Decline of Dry Zone Civilization',
        content: `The decline of dry zone civilization in the 13th century marked a fundamental shift in Sri Lankan history. Multiple factors contributed to this transformation. Foreign invasions, particularly by Kalinga Magha in 1215 CE, caused massive destruction of irrigation systems and social infrastructure. Without maintenance, complex irrigation networks deteriorated, leading to reduced agricultural productivity. Malaria epidemics, possibly exacerbated by disrupted water management, made the dry zone increasingly uninhabitable. Political instability and succession disputes weakened central authority's ability to maintain the elaborate hydraulic systems that sustained civilization. The breakdown of trade networks reduced economic vitality. Gradually, population centers shifted southward and westward to the wet zone where agriculture could be sustained by natural rainfall without extensive irrigation.`,
        duration: '160',
      },
      {
        id: 24,
        title: 'Dambadeniya Kingdom (1220-1345)',
        content: `Dambadeniya became the capital after King Vijayabahu III moved the seat of power from Polonnaruwa around 1220 CE. Located in the western province, Dambadeniya was easier to defend and less vulnerable to South Indian invasions. King Parakramabahu II (1236-1270) was Dambadeniya's most notable ruler, repelling a Pandyan invasion and restoring Buddhist institutions. He was a great patron of literature and the arts, himself authoring several literary works including grammar texts. During this period, the sacred Tooth Relic became increasingly important as a symbol of sovereignty. Dambadeniya featured a palace complex, temples, and defensive walls. However, the kingdom's political stability was short-lived, facing continuous pressure from both internal disputes and external threats.`,
        duration: '145',
      },
      {
        id: 25,
        title: 'Yapahuwa Kingdom (1272-1293)',
        content: `Yapahuwa served as the capital for a brief but significant period under King Bhuvanekabahu I. The kingdom was centered on a massive rock fortress rising 90 meters above the surrounding plains. Yapahuwa's most distinctive feature was its magnificent stone stairway with elaborate carvings and a temple at the summit that housed the sacred Tooth Relic. The architectural style showed South Indian influence, particularly in decorative elements. The fortress was designed to be impregnable, with multiple defensive levels. However, in 1284 CE, a South Indian invasion successfully captured Yapahuwa and took the Tooth Relic to India, from where it was later recovered. After this traumatic event, the capital was moved again, and Yapahuwa was abandoned, leaving behind its impressive rock fortress as a monument.`,
        duration: '145',
      },
      {
        id: 26,
        title: 'Kurunegala Kingdom (1293-1341)',
        content: `Kurunegala became the capital after Yapahuwa's fall, chosen for its strategic location in the central region. The city was surrounded by distinctive rock formations, including Ethagala (Elephant Rock), which served as natural defenses. King Parakramabahu IV made Kurunegala a significant religious and administrative center, though details of this period are less documented than earlier eras. The kingdom faced continuous challenges from South Indian invasions and internal political divisions. Despite these difficulties, trade and commerce continued, with Arab and Chinese traders visiting Sri Lankan ports. The period saw continued Buddhist cultural activities, though on a smaller scale than during Anuradhapura and Polonnaruwa. By the mid-14th century, the capital shifted again to Gampola.`,
        duration: '135',
      },
      {
        id: 27,
        title: 'Gampola Kingdom (1341-1408)',
        content: `Gampola, located in the central highlands, became the capital during a period of political fragmentation. King Bhuvanekabahu IV established Gampola as his capital, seeking security in the mountainous terrain. The kingdom is notable for the Gadaladeniya and Lankatilaka temples, which showcase a unique architectural style blending Sinhalese and South Indian elements. These temples feature stone construction and beautiful frescoes. The Embekka Devalaya, with its intricate wood carvings, represents the finest examples of medieval Sinhalese craftsmanship. During the Gampola period, Sri Lanka was effectively divided into multiple kingdoms, with different rulers controlling various regions. This political division would continue and intensify, eventually leading to the establishment of the Kotte kingdom in the lowlands.`,
        duration: '140',
      },
    ],
  },
  {
    id: 7,
    title: 'Kotte & Kandy Kingdoms',
    description: 'Learn about the last major Sinhalese kingdoms before and during colonization',
    duration: '32 min',
    topics: [
      {
        id: 28,
        title: 'Kotte Kingdom - Foundation and Rise',
        content: `The Kingdom of Kotte was established around 1371 CE by King Alakesvara near present-day Colombo. The choice of location in the wet zone lowlands reflected the permanent shift from dry zone civilization. Kotte's strategic position near the coast facilitated maritime trade, which was becoming increasingly important. The city was heavily fortified with a moat and ramparts. Under King Parakramabahu VI (1412-1467), Kotte reached its zenith, becoming the most powerful kingdom in Sri Lanka. Parakramabahu VI unified the entire island under his rule, the last king to achieve this feat. His reign saw a golden age of Sinhalese literature, with the composition of major works like the Salalihini Sandesaya. Kotte became a center of Buddhist learning and cultural achievement.`,
        duration: '155',
      },
      {
        id: 29,
        title: 'Parakramabahu VI - The Last Great King',
        content: `King Parakramabahu VI (1412-1467) was the last ruler to unify Sri Lanka under a single crown. His 55-year reign brought unprecedented cultural and literary development. He was a great patron of arts and literature, with his court supporting renowned poets and scholars. The Salalihini Sandesaya, Gira Sandesaya, and other literary masterpieces were composed during his time. He maintained diplomatic relations with foreign powers, including China, and promoted international trade. Parakramabahu VI reformed Buddhist institutions and sponsored the construction and renovation of temples. His administrative system was efficient, though after his death the kingdom fragmented. His reign represented the last flowering of indigenous Sinhalese political power before the arrival of European colonial forces.`,
        duration: '150',
      },
      {
        id: 30,
        title: 'Kandyan Kingdom - Origins and Significance',
        content: `The Kandyan Kingdom emerged in the central highlands during the 15th century, initially as a subordinate kingdom to Kotte. Its mountainous location provided natural defenses that would prove crucial as European powers colonized the coastal areas. After Kotte's fragmentation following Parakramabahu VI's death, Kandy became increasingly independent. King Vimaladharmasuriya I (1592-1604) consolidated Kandyan power and successfully resisted Portuguese attempts at conquest. The kingdom's significance grew as it became the last independent Sinhalese kingdom, preserving traditional culture and Buddhism while coastal regions fell under Portuguese and later Dutch control. Kandy's survival for over three centuries of European colonialism made it a symbol of Sinhalese resistance and cultural continuity.`,
        duration: '145',
      },
      {
        id: 31,
        title: 'Kandyan Administration and Society',
        content: `The Kandyan Kingdom developed a sophisticated administrative system adapted to its mountainous terrain. The kingdom was divided into provinces (disavas), each governed by a disava appointed by the king. Below were smaller administrative units (korales and pattus). The radala aristocracy formed the ruling class, holding hereditary positions and controlling lands. Society was hierarchically organized with distinct castes, each with specific occupations and social roles. The govigama (cultivators) were the highest caste, followed by artisan and service castes. The system of rajakariya (obligatory service to the king) was central to administration, with different castes providing specific services. Land tenure was complex, with various categories of rights and obligations.`,
        duration: '145',
      },
      {
        id: 32,
        title: 'Kandyan Culture and Foreign Relations',
        content: `Kandyan culture preserved and developed traditional Sinhalese arts, crafts, and customs. The Temple of the Tooth Relic in Kandy became the kingdom's spiritual and political center, with the Esala Perahera festival demonstrating the kingdom's grandeur. Arts flourished, including dance, music, and craftsmanship. The kingdom maintained diplomatic relations with various powers, playing European nations against each other to preserve independence. Kings corresponded with Portuguese, Dutch, and British authorities, making treaties while resisting subjugation. The kingdom also maintained relations with Asian powers, including India and Southeast Asian kingdoms. This diplomatic skill, combined with mountainous defenses, enabled Kandy to remain independent until 1815.`,
        duration: '140',
      },
    ],
  },
  {
    id: 8,
    title: 'European Colonization - Portuguese Period',
    description: 'Study the arrival of Portuguese and their impact on Sri Lankan society',
    duration: '28 min',
    topics: [
      {
        id: 33,
        title: 'Portuguese Arrival (1505)',
        content: `The Portuguese arrived in Sri Lanka in 1505 when a fleet commanded by Lourenço de Almeida was blown off course and reached Colombo. Initially received as traders, the Portuguese quickly established their first fort in Colombo in 1517. Their arrival marked the beginning of European colonialism in Sri Lanka. The Portuguese were primarily interested in the cinnamon trade and controlling strategic ports in the Indian Ocean trade network. They employed a strategy of making alliances with local kingdoms while gradually expanding their control. By the mid-16th century, they controlled most coastal areas, though they never conquered the Kandyan Kingdom in the interior. Portuguese rule lasted until 1658 when they were displaced by the Dutch.`,
        duration: '145',
      },
      {
        id: 34,
        title: 'Portuguese Impact on Religion',
        content: `The Portuguese brought Roman Catholicism to Sri Lanka, pursuing an aggressive policy of religious conversion. They established churches, schools, and missionary institutions, particularly in areas under their direct control. Franciscan and Dominican missionaries worked to convert the population, sometimes using forceful methods. Buddhist and Hindu temples were destroyed in some areas, and discriminatory policies favored Catholic converts. Despite persecution, Buddhism and Hinduism survived, especially in the Kandyan Kingdom. However, Portuguese influence created a lasting Catholic community in Sri Lanka, particularly along the coast. The period also saw the blending of Portuguese and local customs, evident in names, language (many Portuguese words entered Sinhala and Tamil), and cultural practices.`,
        duration: '140',
      },
      {
        id: 35,
        title: 'Portuguese Trade and Political Control',
        content: `Portuguese economic interests centered on the cinnamon monopoly, which they ruthlessly enforced. They established a systematic exploitation of this valuable spice, forcing cultivators to provide cinnamon as tribute. The Portuguese also controlled other trade items including pearls, gems, and elephants. They established a fort-based system of control, with fortified settlements at Colombo, Galle, Jaffna, and other strategic points. Politically, they interfered in succession disputes of Sinhalese kingdoms, supporting favorable candidates. They claimed sovereignty over coastal kingdoms while struggling to subdue the interior. Constant conflicts with the Kandyan Kingdom drained Portuguese resources. Their administrative system was primarily focused on extraction of resources rather than development.`,
        duration: '145',
      },
    ],
  },
  {
    id: 9,
    title: 'European Colonization - Dutch Period',
    description: 'Explore Dutch administration, law, and economic activities in Sri Lanka',
    duration: '25 min',
    topics: [
      {
        id: 36,
        title: 'Dutch Arrival and Control (1658-1796)',
        content: `The Dutch arrived in Sri Lanka in 1602 but only gained control in 1658 after forming an alliance with the Kandyan Kingdom against the Portuguese. Initially invited as allies, the Dutch refused to cede coastal areas to Kandy after defeating the Portuguese, establishing their own colonial rule. The Dutch East India Company (VOC) governed Dutch territories, viewing Sri Lanka primarily as a source of commercial profit. They controlled the maritime provinces while the Kandyan Kingdom remained independent in the interior. Dutch rule was more systematic and organized than Portuguese rule, establishing an efficient administrative structure. However, their commercial exploitation was equally intensive, focusing on monopolizing trade in cinnamon, which they cultivated in plantations.`,
        duration: '150',
      },
      {
        id: 37,
        title: 'Dutch Administration and Roman-Dutch Law',
        content: `The Dutch established a sophisticated administrative system dividing territories into provinces (disawanis), with each governed by a Dutch administrator assisted by local officials. They maintained detailed records of population, land ownership, and revenue in documents called thombo registers. One of the most lasting Dutch contributions was the introduction of Roman-Dutch Law, which became the foundation of Sri Lankan common law and is still applied in areas like marriage and property. The Dutch legal system included courts at different levels, with appeals possible to higher authorities. They also developed a police system and established rules for land ownership and transfer. This legal framework brought more systematic governance than the Portuguese period.`,
        duration: '145',
      },
      {
        id: 38,
        title: 'Dutch Economic Activities and Society',
        content: `The Dutch intensified commercial agriculture, expanding cinnamon cultivation and establishing plantations. They held a strict monopoly on cinnamon trade, punishing unauthorized harvesting severely. The Dutch also developed other industries including areca nut cultivation, elephant capture, and gem mining. They improved infrastructure, building roads, bridges, and irrigation works to facilitate trade. The Dutch introduced a more systematic taxation system. Socially, they were less aggressively religious than the Portuguese, though they promoted Protestantism and restricted Catholic worship. The Dutch Reformed Church established schools and churches. Dutch influence is visible in Sri Lankan architecture, legal system, and surnames. They also created detailed maps and documentation of Sri Lanka.`,
        duration: '140',
      },
    ],
  },
  {
    id: 10,
    title: 'British Colonization - Conquest and Early Rule',
    description: 'Learn about British conquest and the transformation of Sri Lankan society',
    duration: '35 min',
    topics: [
      {
        id: 39,
        title: 'British Arrival and Coastal Conquest (1796)',
        content: `The British captured Dutch territories in Sri Lanka in 1796 during the Napoleonic Wars, initially taking control on behalf of the exiled Prince of Orange. After the wars, Britain retained Ceylon (as they called it) as a Crown Colony. Unlike the Portuguese and Dutch, the British already controlled India, making Ceylon strategically important for controlling sea routes to India and the Far East. The British immediately began consolidating control over coastal areas, maintaining the Dutch administrative structure initially. They were more interested in strategic control than immediate commercial profit, though economic exploitation would intensify later. The British also coveted the Kandyan Kingdom, which remained independent until 1815.`,
        duration: '145',
      },
      {
        id: 40,
        title: 'Kandyan Convention (1815)',
        content: `The Kandyan Convention of 1815 marked the end of Sri Lankan independence and brought the entire island under British rule for the first time. The convention was signed on March 2, 1815, between British officials and Kandyan chiefs who had rebelled against King Sri Vikrama Rajasinha. The chiefs, dissatisfied with their king (who was of South Indian origin), invited British intervention. The convention had several key provisions: the king was deposed, Buddhism would be protected, Kandyan law and customs would be preserved, and chiefs would retain their positions under British oversight. However, the British soon violated these provisions, interfering with Buddhist institutions and Kandyan administration. The convention was a defining moment, ending over 2,300 years of independent kingship in Sri Lanka.`,
        duration: '155',
      },
      {
        id: 41,
        title: 'Uva Rebellion (1818)',
        content: `The Uva Rebellion of 1818 was the first major resistance against British rule following the Kandyan Convention. Led by chieftains including Keppetipola Disawe who initially served the British, the rebellion was sparked by British violations of the Kandyan Convention, particularly interference with Buddhist practices and traditional administration. The rebels proclaimed a new king and mobilized armed resistance in the Uva and Wellassa regions. The rebellion spread across the Kandyan provinces, with thousands participating. The British response was brutal: they burned villages, destroyed crops, and executed captured rebels including Keppetipola. The rebellion was suppressed by November 1818, but it cost many lives and caused tremendous destruction. The British thereafter imposed harsher control over the Kandyan provinces.`,
        duration: '150',
      },
      {
        id: 42,
        title: 'Colebrooke-Cameron Reforms (1833)',
        content: `The Colebrooke-Cameron Reforms of 1833 fundamentally restructured the administration and economy of British Ceylon. The Colebrooke Commission recommended abolishing the existing administrative system and creating a unified government for the whole island. Key reforms included: establishing a central government with a Governor, Executive Council, and Legislative Council (which included some nominated local members, introducing limited participation); abolishing the rajakariya (compulsory service) system; introducing private land ownership (previously land was held under various traditional tenure systems); unifying the judicial system; and promoting English education. The Cameron recommendations focused on economic development, suggesting the development of plantation agriculture. These reforms transformed Sri Lankan society, breaking down traditional structures and facilitating colonial economic exploitation.`,
        duration: '160',
      },
    ],
  },
  {
    id: 11,
    title: 'British Colonization - Plantation Economy',
    description: 'Examine the plantation system and its profound impact on Sri Lankan society',
    duration: '30 min',
    topics: [
      {
        id: 43,
        title: 'Coffee Plantations and Their Impact',
        content: `Coffee cultivation began in Ceylon in the early 19th century, initially as small-scale enterprises. After the Colebrooke-Cameron reforms facilitated private land ownership, British planters established extensive coffee plantations in the central highlands. By the 1870s, coffee was Ceylon's major export, with over 250,000 acres under cultivation. The industry attracted British capital and settlers. However, in the 1870s, a devastating coffee leaf disease (Hemileia vastatrix) destroyed the plantations, causing economic catastrophe. The rapid expansion of coffee plantations had required massive labor, leading to the importation of South Indian Tamil laborers. It also resulted in extensive deforestation as hillside forests were cleared. The coffee era transformed the landscape, economy, and demography of the island before collapsing in the late 1870s.`,
        duration: '150',
      },
      {
        id: 44,
        title: 'Tea Plantation Development',
        content: `After the coffee industry collapsed, planters desperately sought alternatives. James Taylor, a Scottish planter, pioneered commercial tea cultivation in Ceylon in 1867 at the Loolecondera estate in Kandy. Following coffee's demise, tea rapidly expanded to replace it. By the 1890s, Ceylon had become a major tea producer. The industry required significant capital investment in processing facilities and infrastructure. Tea cultivation proved well-suited to Ceylon's climate and terrain. The plantation system expanded dramatically, with large estates owned by British companies. The industry created a distinctive social structure with British planters and managers at the top, a layer of skilled workers and supervisors, and a large workforce of Indian Tamil laborers. Tea transformed Ceylon into a major exporter, with "Ceylon Tea" becoming a recognized global brand.`,
        duration: '150',
      },
      {
        id: 45,
        title: 'Indian Tamil Labor Migration',
        content: `The plantation economy required massive labor forces that local populations couldn't or wouldn't provide. From the 1840s onwards, hundreds of thousands of Tamil laborers were brought from South India to work in coffee and later tea plantations. Recruitment was organized through kangani (labor supervisors) who brought workers from their home villages. Working conditions were harsh: long hours, low wages, poor housing in "line rooms," and limited access to healthcare and education. Families lived in plantation estates, creating isolated communities. This migration created a distinct Indian Tamil community in Ceylon, separate from the indigenous Sri Lankan Tamil population. By independence, over one million Indian Tamils lived in Ceylon, mostly in central plantation districts. Their status and rights would become a contentious political issue.`,
        duration: '155',
      },
      {
        id: 46,
        title: 'Rubber and Coconut Industries',
        content: `Alongside tea, rubber and coconut plantations expanded in the late 19th and early 20th centuries. Rubber cultivation began in the 1880s, with plants brought from Brazil via Kew Gardens in England. The industry grew rapidly in the early 20th century driven by automobile industry demand. Rubber plantations were established in the wet zone lowlands. Coconut cultivation, both in small holdings and plantations, expanded along coastal areas. Coconut products including copra, coconut oil, and desiccated coconut became important exports. Unlike tea, coconut cultivation involved many Sinhalese smallholders as well as large estates. These crops diversified Ceylon's economy but maintained its dependence on agricultural exports vulnerable to price fluctuations in global markets.`,
        duration: '145',
      },
    ],
  },
  {
    id: 12,
    title: 'British Colonization - Transport & Education',
    description: 'Study infrastructure development and the transformation of education',
    duration: '25 min',
    topics: [
      {
        id: 47,
        title: 'Railway Development',
        content: `Railway construction in Ceylon began in 1858, with the first line connecting Colombo to Ambepussa opening in 1865. The railway system expanded to serve plantation areas, connecting Colombo to Kandy (1867), and eventually extending to Badulla and other regions. Railways were built primarily to transport plantation products to Colombo port, though they also facilitated passenger travel. Construction was expensive and challenging due to mountainous terrain, requiring impressive engineering including tunnels and viaducts. The railway transformed travel and commerce, making previously remote areas accessible. It facilitated the movement of goods and people, integrated the economy, and contributed to cultural exchange. However, the system was designed to serve colonial economic interests rather than local development needs.`,
        duration: '145',
      },
      {
        id: 48,
        title: 'Road Development and Infrastructure',
        content: `The British developed an extensive road network connecting plantations to ports and administrative centers. Major roads were built or improved during the early colonial period, including routes to Kandy and the central highlands. Road construction employed forced labor initially, though this was abolished after the Colebrooke-Cameron reforms. The British also developed Colombo harbor into a modern port facility, making it a major transshipment point in the Indian Ocean trade network. Telegraph systems were installed, improving communications. Bridge construction enabled year-round transport across rivers. This infrastructure primarily served colonial economic interests - extracting plantation products and importing manufactured goods - but also created a foundation for modern transportation systems.`,
        duration: '140',
      },
      {
        id: 49,
        title: 'English Education System',
        content: `The British introduced English education to create a class of locals who could serve in colonial administration. Christian missionary schools, established from the 1820s onwards, played a central role in spreading English education. The government also established schools and colleges. English education provided access to administrative positions and professional careers, creating a new English-educated elite. However, it was limited to a small minority, mostly from higher social classes in urban areas. The vast majority of the population remained outside this system. English education created social divisions between the English-educated elite and the masses. It also spread Western ideas including liberal political thought, which ironically would fuel nationalist movements demanding independence.`,
        duration: '145',
      },
      {
        id: 50,
        title: 'Vernacular and Monastic Education',
        content: `While English education received government support, traditional vernacular education in Sinhala and Tamil continued, primarily through Buddhist and Hindu religious institutions. Buddhist pirivenas (monastic schools) continued teaching Buddhism, Pali, Sinhala literature, and traditional subjects. Hindu schools taught Tamil and religious texts. However, these institutions faced challenges as government support and social prestige shifted to English schools. A revival of indigenous education occurred in the early 20th century through efforts of religious and nationalist leaders. Buddhist schools like Ananda and Nalanda Colleges combined English education with Buddhist values. This created a dual education system reflecting the broader tension between traditional culture and colonial modernity.`,
        duration: '145',
      },
    ],
  },
  {
    id: 13,
    title: 'National Movement & Path to Independence',
    description: 'Trace the growth of nationalism and the struggle for independence',
    duration: '32 min',
    topics: [
      {
        id: 51,
        title: 'Growth of Nationalism - Early Phase',
        content: `Sri Lankan nationalism emerged in the late 19th century, initially as cultural and religious revival movements. The Buddhist revivalist movement led by figures like Anagarika Dharmapala challenged Christian missionary activities and promoted Buddhist education and culture. The Temperance Movement protested against alcohol consumption promoted by colonial authorities. These movements instilled pride in indigenous culture and history. The formation of organizations like the Young Men's Buddhist Association (YMBA) provided platforms for nationalist activities. Early nationalism was primarily cultural rather than explicitly political, but it created consciousness of a distinct identity and grievances against colonial rule. Similar Hindu and Muslim revivalist movements emerged among Tamil and Muslim communities.`,
        duration: '150',
      },
      {
        id: 52,
        title: 'Political Awakening and Constitutional Reforms',
        content: `Political nationalism developed in the early 20th century. The Ceylon National Congress, founded in 1919, became the first major political organization demanding constitutional reforms and greater representation. Leaders included educated elites who initially sought reform rather than independence. Limited constitutional reforms in 1910, 1920, and 1924 gradually increased local representation in the Legislative Council, though power remained with British officials. The Donoughmore Constitution of 1931 introduced universal adult suffrage, making Ceylon the first Asian colony with universal franchise. This was a significant advancement, though the British retained control over key areas. These reforms created political space for nationalist leaders to organize and articulate demands for self-governance.`,
        duration: '150',
      },
      {
        id: 53,
        title: 'Role of Nationalist Leaders',
        content: `Several leaders played crucial roles in the independence movement. D.S. Senanayake emerged as the principal leader, advocating for independence through constitutional means. He skillfully negotiated with British authorities while building political coalitions. S.W.R.D. Bandaranaike initially worked with Senanayake but later formed the Sri Lanka Freedom Party appealing to Sinhala Buddhist sentiments. Tamil leaders like G.G. Ponnambalam advocated for minority rights within the framework of a united Ceylon. The Lanka Sama Samaja Party, founded by leaders like N.M. Perera and Colvin R. de Silva, promoted socialist ideology and mass mobilization. These diverse leaders reflected different ideological and communal perspectives within the nationalist movement.`,
        duration: '150',
      },
      {
        id: 54,
        title: 'Formation of Political Parties',
        content: `The formation of political parties transformed Sri Lankan politics in the 1930s-1940s. The Ceylon National Congress dominated initially but fragmented as different factions emerged. The Lanka Sama Samaja Party (LSSP), founded in 1935, was the first Marxist party, appealing to workers and the educated middle class with its socialist ideology. The Communist Party formed in 1943. S.W.R.D. Bandaranaike founded the Sinhala Maha Sabha in 1937, which later became the Sri Lanka Freedom Party (SLFP) promoting Sinhala Buddhist nationalism. Tamil political organizations including the Tamil Congress advocated for Tamil rights. This party system reflected ideological diversity and communal divisions that would shape post-independence politics.`,
        duration: '145',
      },
      {
        id: 55,
        title: 'Independence in 1948',
        content: `Ceylon achieved independence on February 4, 1948, through constitutional negotiation rather than armed struggle. World War II weakened Britain's capacity to maintain colonial control, and the Labour government in Britain was willing to grant independence to colonies. D.S. Senanayake's United National Party negotiated the transfer of power. Unlike India, Ceylon's independence was achieved without major violence or partition, though communal tensions existed. The new constitution made Ceylon a dominion within the British Commonwealth, retaining the British monarch as head of state. D.S. Senanayake became the first Prime Minister. Independence was celebrated as a triumph, though challenges including economic development, national integration, and resolving the status of Indian Tamils would test the new nation.`,
        duration: '155',
      },
    ],
  },
  {
    id: 14,
    title: 'Sri Lanka After Independence',
    description: 'Explore political, social, and economic developments in independent Sri Lanka',
    duration: '28 min',
    topics: [
      {
        id: 56,
        title: 'Early Political Development (1948-1956)',
        content: `The first years of independence were dominated by the United National Party (UNP) led by D.S. Senanayake and after his death in 1952, by his son Dudley Senanayake and then Sir John Kotelawala. The government pursued conservative economic policies, maintaining close ties with Britain and the Commonwealth. A controversial issue was citizenship for Indian Tamils: the Ceylon Citizenship Act of 1948 and subsequent legislation disenfranchised most Indian Tamils, creating lasting grievances. The government promoted agriculture and rural development while maintaining the plantation economy. English remained the language of administration and higher education. The UNP represented primarily the English-educated urban elite and rural landlords.`,
        duration: '145',
      },
      {
        id: 57,
        title: 'The 1956 Revolution',
        content: `The 1956 election brought a dramatic political shift. S.W.R.D. Bandaranaike's Mahajana Eksath Peramuna (People's United Front) won a landslide victory on a platform of Sinhala nationalist and socialist policies. The "Sinhala Only" policy made Sinhala the sole official language, marginalizing Tamils and contributing to ethnic tensions. The government promoted state intervention in the economy, nationalizing key sectors. Buddhist cultural revival was encouraged. These policies appealed to the Sinhala-speaking rural majority who felt excluded by English-educated elites. However, the language policy exacerbated ethnic divisions between Sinhalese and Tamils. Bandaranaike attempted to moderate through the Bandaranaike-Chelvanayakam Pact of 1957, but faced opposition from Sinhala extremists.`,
        duration: '150',
      },
      {
        id: 58,
        title: 'Social Changes and Welfare Policies',
        content: `Post-independence Sri Lanka developed an extensive welfare system despite limited resources. Free education from primary to university level was maintained and expanded, increasing literacy rates significantly. A free healthcare system provided basic medical services throughout the island, leading to improvements in life expectancy and reduction in infant mortality. Subsidized rice rations helped ensure food security for the poor. These welfare policies were among the most progressive in the developing world. They created high human development indicators despite relatively low per capita income. However, financing these programs strained government budgets. The welfare policies also created a literate, politically conscious population with high expectations that governments struggled to meet.`,
        duration: '145',
      },
      {
        id: 59,
        title: 'Economic Development Challenges',
        content: `Independent Sri Lanka faced significant economic challenges. The economy remained dependent on plantation exports (tea, rubber, coconut) vulnerable to price fluctuations. Import substitution industrialization was attempted with mixed success. Land reforms were implemented, including the Paddy Lands Act providing security to tenant farmers. The government took over British-owned plantations in the 1970s. However, economic growth was often sluggish, failing to create sufficient employment for the expanding educated population. Youth unemployment became a serious problem, contributing to political instability. The ethnic conflict that escalated from the 1980s further damaged economic development, diverting resources to military spending and disrupting productive activities.`,
        duration: '145',
      },
      {
        id: 60,
        title: 'Education Expansion and Language Politics',
        content: `Education expanded rapidly after independence with free education at all levels. Literacy rates rose significantly, and university education became accessible to talented students from all backgrounds regardless of economic status. However, education became entangled with ethnic politics. The "Sinhala Only" policy affected education, with Tamil students disadvantaged in university admissions conducted in Sinhala. Standardization policies in the 1970s introduced different cutoff marks for different language communities, attempting to balance ethnic representation but creating new grievances. The language of instruction controversy reflected broader ethnic tensions. Despite challenges, Sri Lanka's investment in education created a highly literate population and served as a foundation for human development.`,
        duration: '150',
      },
    ],
  },
];

// Helper function to get lesson by ID
export const getLessonById = (lessonId: number): Lesson | undefined => {
  const allLessons = [...grade10Lessons, ...grade11Lessons];
  return allLessons.find((lesson) => lesson.id === lessonId);
};

// Helper function to get all lessons for a grade
export const getLessonsByGrade = (grade: number): Lesson[] => {
  return grade === 10 ? grade10Lessons : grade11Lessons;
};
