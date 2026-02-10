# Sample lesson data for Grade 10 and Grade 11 History
LESSON_DATA = {
    10: [
        {
            "id": "grade10_lesson1",
            "title": "Ancient Civilizations",
            "description": "Learn about the great ancient civilizations of the world",
            "grade": 10,
            "thumbnail": "[BUILDING]",
            "subsections": [
                {
                    "id": "ancient_egypt",
                    "title": "Ancient Egypt",
                    "duration": 8,
                    "description": "Discover the wonders of Ancient Egypt, the Pharaohs, and pyramids",
                    "content": "Ancient Egypt was one of the world's greatest civilizations. It developed along the Nile River and lasted for over 3,000 years. The Egyptians built massive pyramids as tombs for their pharaohs. They invented hieroglyphics, a system of writing using pictures and symbols. The civilization was famous for its advances in mathematics, medicine, and architecture."
                },
                {
                    "id": "ancient_greece",
                    "title": "Ancient Greece",
                    "duration": 7,
                    "description": "Explore the birthplace of democracy and Western philosophy",
                    "content": "Ancient Greece gave the world democracy, philosophy, and the Olympic Games. Greek city-states like Athens and Sparta developed different forms of government. The Greeks made advances in mathematics, science, and literature. Famous philosophers like Socrates, Plato, and Aristotle taught in Athens. Greek art and architecture influenced civilizations for centuries to come."
                },
                {
                    "id": "roman_empire",
                    "title": "Roman Empire",
                    "duration": 9,
                    "description": "Discover the rise and fall of the mighty Roman Empire",
                    "content": "The Roman Empire was the largest and most powerful empire of the ancient world. Romans built an incredible network of roads and aqueducts. They created a sophisticated legal system that influenced modern law. The empire stretched across three continents. Romans were skilled engineers who built bridges, roads, and cities that lasted for centuries."
                }
            ]
        },
        {
            "id": "grade10_lesson2",
            "title": "Medieval Period",
            "description": "Explore the fascinating medieval era in Europe",
            "grade": 10,
            "thumbnail": "[CASTLE]",
            "subsections": [
                {
                    "id": "feudalism",
                    "title": "Feudalism",
                    "duration": 8,
                    "description": "Understanding the feudal system and social hierarchy",
                    "content": "Feudalism was the dominant social and economic system in medieval Europe. It was based on the exchange of land for loyalty and military service. The king granted land to nobles called lords. Lords divided their land among knights and peasants called serfs. Serfs worked the land in exchange for protection and a place to live."
                },
                {
                    "id": "knights_castles",
                    "title": "Knights and Castles",
                    "duration": 7,
                    "description": "Learn about medieval knights and their fortified castles",
                    "content": "Medieval knights were skilled warriors who served as the military force of feudalism. They followed a code of conduct called chivalry. Knights lived in castles, which were fortified structures built for defense. Castles had high walls, towers, and moats to protect against enemy attacks. Knights trained from childhood to become expert warriors in combat and horsemanship."
                }
            ]
        }
    ],
    11: [
        {
            "id": "grade11_lesson1",
            "title": "Age of Exploration",
            "description": "Journey through the Age of Exploration and global discovery",
            "grade": 11,
            "thumbnail": "[COMPASS]",
            "subsections": [
                {
                    "id": "exploration_motives",
                    "title": "Motives for Exploration",
                    "duration": 10,
                    "description": "Understanding why Europeans explored the world",
                    "content": "During the Age of Exploration, European nations sent explorers across the oceans. They were motivated by the desire for wealth, spices, and trade. Better ships and navigation tools made long voyages possible. Countries competed to find new trade routes to Asia. The exploration led to the discovery of the Americas and opened new trade networks."
                },
                {
                    "id": "famous_explorers",
                    "title": "Famous Explorers",
                    "duration": 9,
                    "description": "Learn about the great explorers like Columbus and da Gama",
                    "content": "Christopher Columbus sailed across the Atlantic Ocean in 1492 seeking a route to Asia. He reached the Americas instead, changing world history. Vasco da Gama sailed around Africa to India, opening a sea route to Asia. Ferdinand Magellan's expedition was the first to circumnavigate the globe. These explorers opened new trade routes and led to increased contact between continents."
                },
                {
                    "id": "colonial_impact",
                    "title": "Colonial Impact",
                    "duration": 11,
                    "description": "The consequences of European colonization on indigenous peoples",
                    "content": "European exploration led to colonization of the Americas, Africa, and Asia. European settlers established colonies to extract resources and expand their empires. The contact between Europeans and indigenous peoples had devastating effects. Diseases brought by Europeans killed millions of indigenous people. This period marked the beginning of the modern world system based on European dominance."
                }
            ]
        },
        {
            "id": "grade11_lesson2",
            "title": "Industrial Revolution",
            "description": "Understand the transformative Industrial Revolution",
            "grade": 11,
            "thumbnail": "[GEAR]",
            "subsections": [
                {
                    "id": "industrial_start",
                    "title": "Start of Industrial Revolution",
                    "duration": 10,
                    "description": "How the Industrial Revolution began in Britain",
                    "content": "The Industrial Revolution began in Britain in the late 1700s and early 1800s. Advances in technology transformed manufacturing and agriculture. The steam engine, developed by James Watt, powered factories and transportation. Coal mining expanded to fuel the steam engines. The textile industry was one of the first to industrialize, with new machines speeding up production. Factories grew larger and more efficient, changing how goods were produced."
                }
            ]
        }
    ]
}


def get_lessons_by_grade(grade: int):
    """Get all lessons for a specific grade"""
    return LESSON_DATA.get(grade, [])


def get_lesson_by_id(grade: int, lesson_id: str):
    """Get a specific lesson by ID"""
    lessons = get_lessons_by_grade(grade)
    for lesson in lessons:
        if lesson['id'] == lesson_id:
            return lesson
    return None


def get_subsection_by_id(grade: int, lesson_id: str, subsection_id: str):
    """Get a specific subsection by ID"""
    lesson = get_lesson_by_id(grade, lesson_id)
    if lesson:
        for subsection in lesson.get('subsections', []):
            if subsection['id'] == subsection_id:
                return subsection
    return None
