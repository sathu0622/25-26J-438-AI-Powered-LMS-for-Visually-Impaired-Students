// Mock Quiz Data for Voice-Enabled Quiz System

export interface QuizQuestion {
  id: number;
  text: string;
  topic: string;
  expectedAnswer: string;
  feedback: string;
}

export interface QuizTopic {
  id: string;
  name: string;
  description: string;
  questions: QuizQuestion[];
}

export const quizTopics: QuizTopic[] = [
  {
    id: 'intro-education',
    name: 'Introduction to Education',
    description: 'Fundamental concepts and purposes of education',
    questions: [
      {
        id: 1,
        text: 'What is the primary purpose of education in society?',
        topic: 'Introduction to Education',
        expectedAnswer: 'The primary purpose of education is to develop individuals intellectually, socially, and emotionally, preparing them to contribute meaningfully to society. Education transmits knowledge, skills, values, and cultural heritage while fostering critical thinking and personal growth.',
        feedback: 'Education serves multiple purposes including knowledge transmission, skill development, socialization, and personal development. It prepares individuals for productive participation in society while promoting critical thinking and lifelong learning.',
      },
      {
        id: 2,
        text: 'How does education contribute to social development?',
        topic: 'Introduction to Education',
        expectedAnswer: 'Education contributes to social development by reducing poverty, promoting equality, fostering social cohesion, and developing human capital. It enables social mobility and creates informed citizens who can participate in democratic processes.',
        feedback: 'Education is a key driver of social development. It breaks cycles of poverty, promotes social equity, and builds the human capital necessary for economic and social progress. Educated populations are better equipped to address social challenges.',
      },
    ],
  },
  {
    id: 'learning-theories',
    name: 'Learning Theories',
    description: 'Major theories explaining how people learn',
    questions: [
      {
        id: 3,
        text: 'Explain the key principles of constructivist learning theory.',
        topic: 'Learning Theories',
        expectedAnswer: 'Constructivism holds that learners actively construct knowledge rather than passively receiving it. Key principles include: learning is an active process, knowledge is built on prior experiences, social interaction facilitates learning, and learning is most effective when situated in meaningful contexts.',
        feedback: 'Constructivist theory, developed by theorists like Piaget and Vygotsky, emphasizes that learners build their own understanding through experience and reflection. This approach contrasts with behaviorist views of learning as passive reception.',
      },
      {
        id: 4,
        text: 'How does behaviorist theory explain the learning process?',
        topic: 'Learning Theories',
        expectedAnswer: 'Behaviorist theory explains learning as a change in observable behavior resulting from environmental stimuli. Learning occurs through conditioning - both classical and operant - where behaviors are shaped by reinforcement and punishment. This theory focuses on measurable behaviors rather than internal mental states.',
        feedback: 'Behaviorism, associated with Pavlov, Watson, and Skinner, views learning as stimulus-response associations. While criticized for ignoring mental processes, behaviorist principles remain valuable for understanding habit formation and skill development.',
      },
    ],
  },
  {
    id: 'inclusive-education',
    name: 'Inclusive Education',
    description: 'Principles and practices of education for all learners',
    questions: [
      {
        id: 5,
        text: 'What are the main benefits of inclusive education for students with visual impairments?',
        topic: 'Inclusive Education',
        expectedAnswer: 'Inclusive education benefits students with visual impairments by providing equal access to learning opportunities, promoting social integration, developing independence and self-advocacy skills, and preparing students for inclusive communities. It also benefits all students by fostering empathy, diversity appreciation, and collaborative skills.',
        feedback: 'Inclusive education creates environments where all students learn together with appropriate supports. For visually impaired students, this means access to adaptive technologies, specialized instruction, and peer interactions that prepare them for inclusive society.',
      },
      {
        id: 6,
        text: 'Describe key principles of Universal Design for Learning (UDL).',
        topic: 'Inclusive Education',
        expectedAnswer: 'Universal Design for Learning has three main principles: provide multiple means of representation (presenting information in various formats), multiple means of action and expression (allowing students to demonstrate learning in different ways), and multiple means of engagement (offering various ways to motivate and engage learners).',
        feedback: 'UDL, developed by CAST, provides a framework for creating flexible learning environments that accommodate individual learning differences. By building in flexibility from the start, UDL reduces barriers and supports all learners.',
      },
    ],
  },
  {
    id: 'assessment-methods',
    name: 'Assessment Methods',
    description: 'Various approaches to evaluating student learning',
    questions: [
      {
        id: 7,
        text: 'What is the difference between formative and summative assessment?',
        topic: 'Assessment Methods',
        expectedAnswer: 'Formative assessment occurs during the learning process to monitor progress and provide feedback for improvement. It is ongoing and developmental. Summative assessment occurs at the end of an instructional period to evaluate overall achievement against standards. Formative is for learning; summative is of learning.',
        feedback: 'Understanding this distinction is crucial for effective teaching. Formative assessments (quizzes, observations, discussions) guide instruction, while summative assessments (final exams, projects) measure achievement. Both are important but serve different purposes.',
      },
      {
        id: 8,
        text: 'How can assessment be made accessible for students with disabilities?',
        topic: 'Assessment Methods',
        expectedAnswer: 'Accessible assessment includes providing accommodations like extended time, alternative formats (braille, large print, audio), assistive technologies, quiet testing environments, and alternative means of demonstrating knowledge. The goal is to assess what students know, not their disability.',
        feedback: 'Fair assessment requires removing barriers that prevent students from showing their knowledge. Accommodations level the playing field without lowering standards. This aligns with principles of inclusive education and equity.',
      },
    ],
  },
  {
    id: 'educational-technology',
    name: 'Educational Technology',
    description: 'Role of technology in modern education',
    questions: [
      {
        id: 9,
        text: 'Describe the role of adaptive technologies in modern education.',
        topic: 'Educational Technology',
        expectedAnswer: 'Adaptive technologies enable students with disabilities to access educational content and demonstrate learning. Examples include screen readers, voice recognition software, braille displays, and alternative input devices. These technologies remove barriers and create equal access to education.',
        feedback: 'Adaptive technologies are essential for inclusive education. They transform how students with disabilities can participate in learning. Modern technologies like AI-powered screen readers and voice interfaces continue to expand possibilities for accessibility.',
      },
      {
        id: 10,
        text: 'What are the benefits and challenges of online learning?',
        topic: 'Educational Technology',
        expectedAnswer: 'Benefits include flexibility, accessibility from anywhere, personalized pacing, and access to diverse resources. Challenges include digital divide issues, lack of face-to-face interaction, need for self-discipline, and potential accessibility barriers for students with disabilities.',
        feedback: 'Online learning has expanded dramatically, especially since 2020. While it offers unprecedented flexibility and access, educators must address equity issues and ensure online environments are accessible to all learners.',
      },
    ],
  },
  {
    id: 'special-education',
    name: 'Special Education',
    description: 'Specialized instruction for diverse learning needs',
    questions: [
      {
        id: 11,
        text: 'What is an Individualized Education Program (IEP)?',
        topic: 'Special Education',
        expectedAnswer: 'An IEP is a written document developed for each student with disabilities that outlines specific learning goals, necessary accommodations and modifications, specialized services, and measures of progress. It is created collaboratively by educators, parents, and specialists.',
        feedback: 'IEPs are foundational to special education, ensuring students receive individualized support based on their unique needs. The collaborative development process ensures all stakeholders contribute to the student\'s educational plan.',
      },
      {
        id: 12,
        text: 'How do multi-sensory learning approaches enhance student engagement?',
        topic: 'Special Education',
        expectedAnswer: 'Multi-sensory approaches engage multiple senses simultaneously (visual, auditory, kinesthetic, tactile), reinforcing learning through different pathways. This benefits all learners but is especially effective for students with learning differences, ADHD, or sensory impairments by providing multiple access points to information.',
        feedback: 'Multi-sensory instruction recognizes that students learn through different modalities. By engaging multiple senses, educators increase the likelihood that information will be processed, understood, and retained. This approach aligns with Universal Design for Learning principles.',
      },
    ],
  },
];

// Helper functions
export const getTopicByName = (topicName: string): QuizTopic | undefined => {
  return quizTopics.find(
    (topic) => topic.name.toLowerCase() === topicName.toLowerCase()
  );
};

export const getTopicById = (topicId: string): QuizTopic | undefined => {
  return quizTopics.find((topic) => topic.id === topicId);
};

export const getQuestionsByTopic = (topicName: string): QuizQuestion[] => {
  const topic = getTopicByName(topicName);
  return topic?.questions || [];
};

export const getAllTopics = (): string[] => {
  return quizTopics.map((topic) => topic.name);
};
