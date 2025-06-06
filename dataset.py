# generate_dataset.py
import csv
import numpy as np

def create_training_dataset(filename, num_entries=200):
    # Question templates
    question_templates = [
        "What is {subject}?",
        "How does {subject} work?",
        "Why is {subject} important?",
        "Can you explain {subject}?",
        "When should I use {subject}?",
        "Who created {subject}?",
        "Where can I learn about {subject}?",
        "What are the benefits of {subject}?",
        "How to implement {subject}?",
        "Is {subject} better than {alternative}?"
    ]
    
    # Statement templates
    statement_templates = [
        "I think {subject} is {opinion}.",
        "{subject} seems {adjective}.",
        "Today I worked with {subject}.",
        "{subject} is used for {purpose}.",
        "Let's discuss {subject}.",
        "In my experience, {subject} requires {requirement}.",
        "{subject} has changed {domain}.",
        "Many companies use {subject} for {application}.",
        "The future of {subject} looks {future_adj}.",
        "Learning {subject} helps with {benefit}."
    ]
    
    # Vocabulary lists
    subjects = ["AI", "machine learning", "Python", "neural networks", 
               "data science", "deep learning", "NLP", "computer vision",
               "big data", "cloud computing"]
    
    modifiers = {
        'opinion': ["revolutionary", "overrated", "essential", "complex",
                   "promising", "challenging", "innovative"],
        'adjective': ["powerful", "complicated", "versatile", "disruptive",
                      "advanced", "sophisticated"],
        'purpose': ["data analysis", "automation", "predictions", 
                   "pattern recognition", "optimization"],
        'alternative': ["traditional methods", "manual processes", "older systems"],
        'requirement': ["careful planning", "large datasets", "computational resources"],
        'domain': ["technology", "business", "healthcare", "education"],
        'application': ["decision making", "process optimization", "customer insights"],
        'future_adj': ["bright", "uncertain", "promising", "competitive"],
        'benefit': ["problem solving", "career growth", "research", "innovation"]
    }

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])  # Header row
        
        for _ in range(num_entries):
            if np.random.rand() > 0.5:  # Generate question
                template = np.random.choice(question_templates)
                subject = np.random.choice(subjects)
                
                if "{alternative}" in template:
                    alternative = np.random.choice(modifiers['alternative'])
                    text = template.format(subject=subject, alternative=alternative)
                else:
                    text = template.format(subject=subject)
                    
                label = "Question"
            else:  # Generate statement
                template = np.random.choice(statement_templates)
                subject = np.random.choice(subjects)
                text = template.format(
                    subject=subject,
                    opinion=np.random.choice(modifiers['opinion']),
                    adjective=np.random.choice(modifiers['adjective']),
                    purpose=np.random.choice(modifiers['purpose']),
                    requirement=np.random.choice(modifiers['requirement']),
                    domain=np.random.choice(modifiers['domain']),
                    application=np.random.choice(modifiers['application']),
                    future_adj=np.random.choice(modifiers['future_adj']),
                    benefit=np.random.choice(modifiers['benefit'])
                )
                label = "Statement"
            
            writer.writerow([text, label])

if __name__ == "__main__":
    create_training_dataset("training_dataset.csv", 200)
    print("Dataset generated successfully: training_dataset.csv")
