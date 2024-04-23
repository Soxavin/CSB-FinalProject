# CSB-FinalProject

Farming Advice Personalization System

Project Introduction:
The "Farming Advice Personalization System" is designed to provide farmers with specific, actionable advice on how to optimize the conditions for various crops based on ideal parameters. The system processes user-uploaded data about current farm conditions, compares it with ideal conditions, and outputs personalized recommendations to improve crop yield.

Problem Solved:
This system tackles the challenge of agricultural inefficiencies related to suboptimal farming practices. By comparing current farm conditions against scientifically researched ideal conditions for various crops, it provides tailored advice that can help farmers maximize yields, reduce waste, and improve sustainability.

References:
No external references were used. The program was coded from scratch. However, the help of ChatGPT was enlisted.

Concepts, Models, Functions, Algorithms Applied:
-	Data Handling: Use of Pandas for data manipulation and comparison. 
-	Conditional Logic: Application of complex conditional statements to compare actual farm conditions with ideal parameters. 
-	User Interaction: Streamlit library is used for creating an interactive web application that allows users to upload data and view recommendations. 
-	Data Visualization: Streamlit's UI capabilities are used to display data and results in a user-friendly format. 
-	Categorization Algorithm: Implementation of categorization based on similarity scores to classify crops into different yield potential categories. 
-	Recommendation Engine: Development of a detailed recommendation system that provides specific advice based on discrepancies between current and ideal conditions.

Application Result:
The application successfully processes the input data to provide: 
- A detailed comparison between current and ideal crop conditions. 
- Categorization of crops based on predicted yield ('High', 'Moderate', 'Low'). 
- Detailed, actionable recommendations for each crop to achieve ideal growing conditions.

Evaluation/Comparison:
- Complexity: The application involves moderate complexity, particularly in data processing and generating tailored advice. 
- Performance: The application is optimized for fast data processing, leveraging pandas for efficient data manipulation and Streamlit for quick rendering of results. 
- Scalability: Scalable to handle larger datasets or more complex decision-making criteria without significant redesign. 
- Usability: Highly user-friendly interface, providing a seamless experience from data upload to viewing recommendations.
